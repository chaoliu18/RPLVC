# Copyright 2020 InterDigital Communications, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

# pylint: disable=E0611,E0401
from compressai.ans import BufferedRansEncoder, RansDecoder
from compressai.entropy_models import EntropyBottleneck, GaussianConditional
from compressai.layers import GDN, MaskedConv2d
from compressai.layers import *

from .utils import conv, deconv, update_registered_buffers

# pylint: enable=E0611,E0401

"""
__all__ = [
    "CompressionModel",
    "FactorizedPrior",
    "ScaleHyperprior",
    "MeanScaleHyperprior",
    "JointAutoregressiveHierarchicalPriors",
]
"""

class CompressionModel(nn.Module):
    """Base class for constructing an auto-encoder with at least one entropy
    bottleneck module.

    Args:
        entropy_bottleneck_channels (int): Number of channels of the entropy
            bottleneck
    """

    def __init__(self, entropy_bottleneck_channels, init_weights=True):
        super().__init__()
        self.entropy_bottleneck = EntropyBottleneck(entropy_bottleneck_channels)

        if init_weights:
            self._initialize_weights()

    def aux_loss(self):
        """Return the aggregated loss over the auxiliary entropy bottleneck
        module(s).
        """
        aux_loss = sum(
            m.loss() for m in self.modules() if isinstance(m, EntropyBottleneck)
        )
        return aux_loss

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, *args):
        raise NotImplementedError()

    def parameters(self):
        """Returns an iterator over the model parameters."""
        for m in self.children():
            if isinstance(m, EntropyBottleneck):
                continue
            for p in m.parameters():
                yield p

    def aux_parameters(self):
        """
        Returns an iterator over the entropy bottleneck(s) parameters for
        the auxiliary loss.
        """
        for m in self.children():
            if not isinstance(m, EntropyBottleneck):
                continue
            for p in m.parameters():
                yield p

    def update(self, force=False):
        """Updates the entropy bottleneck(s) CDF values.

        Needs to be called once after training to be able to later perform the
        evaluation with an actual entropy coder.

        Args:
            force (bool): overwrite previous values (default: False)

        """
        for m in self.children():
            if not isinstance(m, EntropyBottleneck):
                continue
            m.update(force=force)


class FactorizedPrior(CompressionModel):
    r"""Factorized Prior model from J. Balle, D. Minnen, S. Singh, S.J. Hwang,
    N. Johnston: `"Variational Image Compression with a Scale Hyperprior"
    <https://arxiv.org/abs/1802.01436>`_, Int Conf. on Learning Representations
    (ICLR), 2018.

    Args:
        N (int): Number of channels
        M (int): Number of channels in the expansion layers (last layer of the
            encoder and last layer of the hyperprior decoder)
    """

    def __init__(self, N, M, **kwargs):
        super().__init__(entropy_bottleneck_channels=M, **kwargs)

        self.g_a = nn.Sequential(
            conv(3, N),
            GDN(N),
            conv(N, N),
            GDN(N),
            conv(N, N),
            GDN(N),
            conv(N, M),
        )

        self.g_s = nn.Sequential(
            deconv(M, N),
            GDN(N, inverse=True),
            deconv(N, N),
            GDN(N, inverse=True),
            deconv(N, N),
            GDN(N, inverse=True),
            deconv(N, 3),
        )

        self.N = N
        self.M = M

    @property
    def downsampling_factor(self) -> int:
        return 2 ** 4

    def forward(self, x):
        y = self.g_a(x)
        y_hat, y_likelihoods = self.entropy_bottleneck(y)
        x_hat = self.g_s(y_hat)

        return {
            "x_hat": x_hat,
            "likelihoods": {
                "y": y_likelihoods,
            },
        }

    def load_state_dict(self, state_dict):
        # Dynamically update the entropy bottleneck buffers related to the CDFs
        update_registered_buffers(
            self.entropy_bottleneck,
            "entropy_bottleneck",
            ["_quantized_cdf", "_offset", "_cdf_length"],
            state_dict,
        )
        super().load_state_dict(state_dict)

    @classmethod
    def from_state_dict(cls, state_dict):
        """Return a new model instance from `state_dict`."""
        N = state_dict["g_a.0.weight"].size(0)
        M = state_dict["g_a.6.weight"].size(0)
        net = cls(N, M)
        net.load_state_dict(state_dict)
        return net

    def compress(self, x):
        y = self.g_a(x)
        y_strings = self.entropy_bottleneck.compress(y)
        return {"strings": [y_strings], "shape": y.size()[-2:]}

    def decompress(self, strings, shape):
        assert isinstance(strings, list) and len(strings) == 1
        y_hat = self.entropy_bottleneck.decompress(strings[0], shape)
        x_hat = self.g_s(y_hat)
        return {"x_hat": x_hat}


# From Balle's tensorflow compression examples
SCALES_MIN = 0.11
SCALES_MAX = 256
SCALES_LEVELS = 64


def get_scale_table(
    min=SCALES_MIN, max=SCALES_MAX, levels=SCALES_LEVELS
):  # pylint: disable=W0622
    return torch.exp(torch.linspace(math.log(min), math.log(max), levels))


class ScaleHyperprior(CompressionModel):
    r"""Scale Hyperprior model from J. Balle, D. Minnen, S. Singh, S.J. Hwang,
    N. Johnston: `"Variational Image Compression with a Scale Hyperprior"
    <https://arxiv.org/abs/1802.01436>`_ Int. Conf. on Learning Representations
    (ICLR), 2018.

    Args:
        N (int): Number of channels
        M (int): Number of channels in the expansion layers (last layer of the
            encoder and last layer of the hyperprior decoder)
    """

    def __init__(self, N, M, **kwargs):
        super().__init__(entropy_bottleneck_channels=N, **kwargs)

        self.g_a = nn.Sequential(
            conv(3, N),
            GDN(N),
            conv(N, N),
            GDN(N),
            conv(N, N),
            GDN(N),
            conv(N, M),
        )

        self.g_s = nn.Sequential(
            deconv(M, N),
            GDN(N, inverse=True),
            deconv(N, N),
            GDN(N, inverse=True),
            deconv(N, N),
            GDN(N, inverse=True),
            deconv(N, 3),
        )

        self.h_a = nn.Sequential(
            conv(M, N, stride=1, kernel_size=3),
            nn.ReLU(inplace=True),
            conv(N, N),
            nn.ReLU(inplace=True),
            conv(N, N),
        )

        self.h_s = nn.Sequential(
            deconv(N, N),
            nn.ReLU(inplace=True),
            deconv(N, N),
            nn.ReLU(inplace=True),
            conv(N, M, stride=1, kernel_size=3),
            nn.ReLU(inplace=True),
        )

        self.gaussian_conditional = GaussianConditional(None)
        self.N = int(N)
        self.M = int(M)

    @property
    def downsampling_factor(self) -> int:
        return 2 ** (4 + 2)

    def forward(self, x):
        y = self.g_a(x)
        z = self.h_a(torch.abs(y))
        z_hat, z_likelihoods = self.entropy_bottleneck(z)
        scales_hat = self.h_s(z_hat)
        y_hat, y_likelihoods = self.gaussian_conditional(y, scales_hat)
        x_hat = self.g_s(y_hat)

        return {
            "x_hat": x_hat,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
        }

    def load_state_dict(self, state_dict):
        # Dynamically update the entropy bottleneck buffers related to the CDFs
        update_registered_buffers(
            self.entropy_bottleneck,
            "entropy_bottleneck",
            ["_quantized_cdf", "_offset", "_cdf_length"],
            state_dict,
        )
        update_registered_buffers(
            self.gaussian_conditional,
            "gaussian_conditional",
            ["_quantized_cdf", "_offset", "_cdf_length", "scale_table"],
            state_dict,
        )
        super().load_state_dict(state_dict)

    @classmethod
    def from_state_dict(cls, state_dict):
        """Return a new model instance from `state_dict`."""
        N = state_dict["g_a.0.weight"].size(0)
        M = state_dict["g_a.6.weight"].size(0)
        net = cls(N, M)
        net.load_state_dict(state_dict)
        return net

    def update(self, scale_table=None, force=False):
        if scale_table is None:
            scale_table = get_scale_table()
        self.gaussian_conditional.update_scale_table(scale_table, force=force)
        super().update(force=force)

    def compress(self, x):
        y = self.g_a(x)
        z = self.h_a(torch.abs(y))

        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])

        scales_hat = self.h_s(z_hat)
        indexes = self.gaussian_conditional.build_indexes(scales_hat)
        y_strings = self.gaussian_conditional.compress(y, indexes)
        return {"strings": [y_strings, z_strings], "shape": z.size()[-2:]}

    def decompress(self, strings, shape):
        assert isinstance(strings, list) and len(strings) == 2
        z_hat = self.entropy_bottleneck.decompress(strings[1], shape)
        scales_hat = self.h_s(z_hat)
        indexes = self.gaussian_conditional.build_indexes(scales_hat)
        y_hat = self.gaussian_conditional.decompress(strings[0], indexes)
        x_hat = self.g_s(y_hat).clamp_(0, 1)
        return {"x_hat": x_hat}


class MeanScaleHyperprior(ScaleHyperprior):
    r"""Scale Hyperprior with non zero-mean Gaussian conditionals from D.
    Minnen, J. Balle, G.D. Toderici: `"Joint Autoregressive and Hierarchical
    Priors for Learned Image Compression" <https://arxiv.org/abs/1809.02736>`_,
    Adv. in Neural Information Processing Systems 31 (NeurIPS 2018).

    Args:
        N (int): Number of channels
        M (int): Number of channels in the expansion layers (last layer of the
            encoder and last layer of the hyperprior decoder)
    """

    def __init__(self, N, M, **kwargs):
        super().__init__(N, M, **kwargs)

        self.h_a = nn.Sequential(
            conv(M, N, stride=1, kernel_size=3),
            nn.LeakyReLU(inplace=True),
            conv(N, N),
            nn.LeakyReLU(inplace=True),
            conv(N, N),
        )

        self.h_s = nn.Sequential(
            deconv(N, M),
            nn.LeakyReLU(inplace=True),
            deconv(M, M * 3 // 2),
            nn.LeakyReLU(inplace=True),
            conv(M * 3 // 2, M * 2, stride=1, kernel_size=3),
        )

    def forward(self, x):
        y = self.g_a(x)
        z = self.h_a(y)
        z_hat, z_likelihoods = self.entropy_bottleneck(z)
        gaussian_params = self.h_s(z_hat)
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        y_hat, y_likelihoods = self.gaussian_conditional(y, scales_hat, means=means_hat)
        x_hat = self.g_s(y_hat)

        return {
            "x_hat": x_hat,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
        }

    def compress(self, x):
        y = self.g_a(x)
        z = self.h_a(y)

        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])

        gaussian_params = self.h_s(z_hat)
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        indexes = self.gaussian_conditional.build_indexes(scales_hat)
        y_strings = self.gaussian_conditional.compress(y, indexes, means=means_hat)
        return {"strings": [y_strings, z_strings], "shape": z.size()[-2:]}

    def decompress(self, strings, shape):
        assert isinstance(strings, list) and len(strings) == 2
        z_hat = self.entropy_bottleneck.decompress(strings[1], shape)
        gaussian_params = self.h_s(z_hat)
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        indexes = self.gaussian_conditional.build_indexes(scales_hat)
        y_hat = self.gaussian_conditional.decompress(
            strings[0], indexes, means=means_hat
        )
        x_hat = self.g_s(y_hat).clamp_(0, 1)
        return {"x_hat": x_hat}


class JointAutoregressiveHierarchicalPriors(CompressionModel):
    r"""Joint Autoregressive Hierarchical Priors model from D.
    Minnen, J. Balle, G.D. Toderici: `"Joint Autoregressive and Hierarchical
    Priors for Learned Image Compression" <https://arxiv.org/abs/1809.02736>`_,
    Adv. in Neural Information Processing Systems 31 (NeurIPS 2018).

    Args:
        N (int): Number of channels
        M (int): Number of channels in the expansion layers (last layer of the
            encoder and last layer of the hyperprior decoder)
    """

    def __init__(self, N=192, M=192, **kwargs):
        super().__init__(entropy_bottleneck_channels=N, **kwargs)

        self.g_a = nn.Sequential(
            conv(3, N, kernel_size=5, stride=2),
            GDN(N),
            conv(N, N, kernel_size=5, stride=2),
            GDN(N),
            conv(N, N, kernel_size=5, stride=2),
            GDN(N),
            conv(N, M, kernel_size=5, stride=2),
        )

        self.g_s = nn.Sequential(
            deconv(M, N, kernel_size=5, stride=2),
            GDN(N, inverse=True),
            deconv(N, N, kernel_size=5, stride=2),
            GDN(N, inverse=True),
            deconv(N, N, kernel_size=5, stride=2),
            GDN(N, inverse=True),
            deconv(N, 3, kernel_size=5, stride=2),
        )

        self.h_a = nn.Sequential(
            conv(M, N, stride=1, kernel_size=3),
            nn.LeakyReLU(inplace=True),
            conv(N, N, stride=2, kernel_size=5),
            nn.LeakyReLU(inplace=True),
            conv(N, N, stride=2, kernel_size=5),
        )

        self.h_s = nn.Sequential(
            deconv(N, M, stride=2, kernel_size=5),
            nn.LeakyReLU(inplace=True),
            deconv(M, M * 3 // 2, stride=2, kernel_size=5),
            nn.LeakyReLU(inplace=True),
            conv(M * 3 // 2, M * 2, stride=1, kernel_size=3),
        )

        self.entropy_parameters = nn.Sequential(
            nn.Conv2d(M * 12 // 3, M * 10 // 3, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(M * 10 // 3, M * 8 // 3, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(M * 8 // 3, M * 6 // 3, 1),
        )

        self.context_prediction = MaskedConv2d(
            M, 2 * M, kernel_size=5, padding=2, stride=1
        )

        self.gaussian_conditional = GaussianConditional(None)
        self.N = int(N)
        self.M = int(M)

    @property
    def downsampling_factor(self) -> int:
        return 2 ** (4 + 2)

    def forward(self, x):
        y = self.g_a(x)
        z = self.h_a(y)
        z_hat, z_likelihoods = self.entropy_bottleneck(z)
        params = self.h_s(z_hat)

        y_hat = self.gaussian_conditional.quantize(
            y, "noise" if self.training else "dequantize"
        )
        ctx_params = self.context_prediction(y_hat)
        gaussian_params = self.entropy_parameters(
            torch.cat((params, ctx_params), dim=1)
        )
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        _, y_likelihoods = self.gaussian_conditional(y, scales_hat, means=means_hat)
        x_hat = self.g_s(y_hat)

        return {
            "x_hat": x_hat,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
        }

    @classmethod
    def from_state_dict(cls, state_dict):
        """Return a new model instance from `state_dict`."""
        N = state_dict["g_a.0.weight"].size(0)
        M = state_dict["g_a.6.weight"].size(0)
        net = cls(N, M)
        net.load_state_dict(state_dict)
        return net

    def compress(self, x):
        if next(self.parameters()).device != torch.device("cpu"):
            warnings.warn(
                "Inference on GPU is not recommended for the autoregressive "
                "models (the entropy coder is run sequentially on CPU)."
            )

        y = self.g_a(x)
        z = self.h_a(y)

        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])

        params = self.h_s(z_hat)

        s = 4  # scaling factor between z and y
        kernel_size = 5  # context prediction kernel size
        padding = (kernel_size - 1) // 2

        y_height = z_hat.size(2) * s
        y_width = z_hat.size(3) * s

        y_hat = F.pad(y, (padding, padding, padding, padding))

        y_strings = []
        for i in range(y.size(0)):
            string = self._compress_ar(
                y_hat[i : i + 1],
                params[i : i + 1],
                y_height,
                y_width,
                kernel_size,
                padding,
            )
            y_strings.append(string)

        return {"strings": [y_strings, z_strings], "shape": z.size()[-2:]}

    def _compress_ar(self, y_hat, params, height, width, kernel_size, padding):
        cdf = self.gaussian_conditional.quantized_cdf.tolist()
        cdf_lengths = self.gaussian_conditional.cdf_length.tolist()
        offsets = self.gaussian_conditional.offset.tolist()

        encoder = BufferedRansEncoder()
        symbols_list = []
        indexes_list = []

        # Warning, this is slow...
        # TODO: profile the calls to the bindings...
        for h in range(height):
            for w in range(width):
                y_crop = y_hat[:, :, h : h + kernel_size, w : w + kernel_size]
                ctx_p = F.conv2d(
                    y_crop,
                    self.context_prediction.weight,
                    bias=self.context_prediction.bias,
                )

                # 1x1 conv for the entropy parameters prediction network, so
                # we only keep the elements in the "center"
                p = params[:, :, h : h + 1, w : w + 1]
                gaussian_params = self.entropy_parameters(torch.cat((p, ctx_p), dim=1))
                gaussian_params = gaussian_params.squeeze(3).squeeze(2)
                scales_hat, means_hat = gaussian_params.chunk(2, 1)

                indexes = self.gaussian_conditional.build_indexes(scales_hat)

                y_crop = y_crop[:, :, padding, padding]
                y_q = self.gaussian_conditional.quantize(y_crop, "symbols", means_hat)
                y_hat[:, :, h + padding, w + padding] = y_q + means_hat

                symbols_list.extend(y_q.squeeze().tolist())
                indexes_list.extend(indexes.squeeze().tolist())

        encoder.encode_with_indexes(
            symbols_list, indexes_list, cdf, cdf_lengths, offsets
        )

        string = encoder.flush()
        return string

    def decompress(self, strings, shape):
        assert isinstance(strings, list) and len(strings) == 2

        if next(self.parameters()).device != torch.device("cpu"):
            warnings.warn(
                "Inference on GPU is not recommended for the autoregressive "
                "models (the entropy coder is run sequentially on CPU)."
            )

        # FIXME: we don't respect the default entropy coder and directly call the
        # range ANS decoder

        z_hat = self.entropy_bottleneck.decompress(strings[1], shape)
        params = self.h_s(z_hat)

        s = 4  # scaling factor between z and y
        kernel_size = 5  # context prediction kernel size
        padding = (kernel_size - 1) // 2

        y_height = z_hat.size(2) * s
        y_width = z_hat.size(3) * s

        # initialize y_hat to zeros, and pad it so we can directly work with
        # sub-tensors of size (N, C, kernel size, kernel_size)
        y_hat = torch.zeros(
            (z_hat.size(0), self.M, y_height + 2 * padding, y_width + 2 * padding),
            device=z_hat.device,
        )

        for i, y_string in enumerate(strings[0]):
            self._decompress_ar(
                y_string,
                y_hat[i : i + 1],
                params[i : i + 1],
                y_height,
                y_width,
                kernel_size,
                padding,
            )

        y_hat = F.pad(y_hat, (-padding, -padding, -padding, -padding))
        x_hat = self.g_s(y_hat).clamp_(0, 1)
        return {"x_hat": x_hat}

    def _decompress_ar(
        self, y_string, y_hat, params, height, width, kernel_size, padding
    ):
        cdf = self.gaussian_conditional.quantized_cdf.tolist()
        cdf_lengths = self.gaussian_conditional.cdf_length.tolist()
        offsets = self.gaussian_conditional.offset.tolist()

        decoder = RansDecoder()
        decoder.set_stream(y_string)

        # Warning: this is slow due to the auto-regressive nature of the
        # decoding... See more recent publication where they use an
        # auto-regressive module on chunks of channels for faster decoding...
        for h in range(height):
            for w in range(width):
                # only perform the 5x5 convolution on a cropped tensor
                # centered in (h, w)
                y_crop = y_hat[:, :, h : h + kernel_size, w : w + kernel_size]
                ctx_p = F.conv2d(
                    y_crop,
                    self.context_prediction.weight,
                    bias=self.context_prediction.bias,
                )
                # 1x1 conv for the entropy parameters prediction network, so
                # we only keep the elements in the "center"
                p = params[:, :, h : h + 1, w : w + 1]
                gaussian_params = self.entropy_parameters(torch.cat((p, ctx_p), dim=1))
                scales_hat, means_hat = gaussian_params.chunk(2, 1)

                indexes = self.gaussian_conditional.build_indexes(scales_hat)
                rv = decoder.decode_stream(
                    indexes.squeeze().tolist(), cdf, cdf_lengths, offsets
                )
                rv = torch.Tensor(rv).reshape(1, -1, 1, 1)
                rv = self.gaussian_conditional.dequantize(rv, means_hat)

                hp = h + padding
                wp = w + padding
                y_hat[:, :, hp : hp + 1, wp : wp + 1] = rv

    def update(self, scale_table=None, force=False):
        if scale_table is None:
            scale_table = get_scale_table()
        self.gaussian_conditional.update_scale_table(scale_table, force=force)
        super().update(force=force)

    def load_state_dict(self, state_dict):
        # Dynamically update the entropy bottleneck buffers related to the CDFs
        update_registered_buffers(
            self.entropy_bottleneck,
            "entropy_bottleneck",
            ["_quantized_cdf", "_offset", "_cdf_length"],
            state_dict,
        )
        update_registered_buffers(
            self.gaussian_conditional,
            "gaussian_conditional",
            ["_quantized_cdf", "_offset", "_cdf_length", "scale_table"],
            state_dict,
        )
        super().load_state_dict(state_dict)
    
def warp(x, flo):
    B, C, H, W = x.size()
    # mesh grid 
    xx = torch.arange(0, W).view(1,-1).repeat(H,1)
    yy = torch.arange(0, H).view(-1,1).repeat(1,W)
    xx = xx.view(1,1,H,W).repeat(B,1,1,1)
    yy = yy.view(1,1,H,W).repeat(B,1,1,1)
    grid = torch.cat((xx,yy),1).float()

    if x.is_cuda:
        grid = grid.cuda()
    vgrid = Variable(grid) + flo

    # scale grid to [-1,1] 
    vgrid[:,0,:,:] = 2.0*vgrid[:,0,:,:].clone() / max(W-1,1)-1.0
    vgrid[:,1,:,:] = 2.0*vgrid[:,1,:,:].clone() / max(H-1,1)-1.0

    vgrid = vgrid.permute(0,2,3,1)        
    output = nn.functional.grid_sample(x, vgrid, mode='bilinear', padding_mode = "border")
    return output

class FPN(nn.Module):
    def __init__(self, N_in, N_out, N_mid=64):
        super(FPN, self).__init__()
        # Encoder layers
        self.e_layer1 = nn.Sequential(
            ResidualBlockWithStride(N_in, N_mid, stride=2),
            ResidualBlock(N_mid, N_mid),
            )
        self.e_layer2 = nn.Sequential(
            ResidualBlockWithStride(N_mid, N_mid*2, stride=2),
            ResidualBlock(N_mid*2, N_mid*2),
            )
        self.e_layer3 = nn.Sequential(
            ResidualBlockWithStride(N_mid*2, N_mid*4, stride=2),
            ResidualBlock(N_mid*4, N_mid*4),
            )

        # Decoder layers
        self.d_layer3 = nn.Sequential(
            ResidualBlock(N_mid*4, N_mid*4),
            ResidualBlockUpsample(N_mid*4, N_mid*2, 2),
            ) 
        self.d_layer2 = nn.Sequential(
            ResidualBlock(N_mid*2, N_mid*2),
            ResidualBlockUpsample(N_mid*2, N_mid, 2),
            )     
        self.d_layer1 = nn.Sequential(
            ResidualBlock(N_mid, N_mid),
            subpel_conv3x3(N_mid, N_out, 2),
            )      
        # Lateral layers
        self.latlayer1 = nn.Conv2d(N_mid, N_mid, kernel_size=1, stride=1, padding=0)
        self.latlayer2 = nn.Conv2d(N_mid*2, N_mid*2, kernel_size=1, stride=1, padding=0)
    def forward(self, x):
        # Bottom-up
        c1 = self.e_layer1(x)
        c2 = self.e_layer2(c1)
        c3 = self.e_layer3(c2)
        # Top-down
        p3 = self.d_layer3(c3) + self.latlayer2(c2)
        p2 = self.d_layer2(p3) + self.latlayer1(c1)
        p1 = self.d_layer1(p2) 
        # Smooth
        return p1

class ResNet(nn.Module):
    def __init__(self, N_in, N_out, N_mid=128, num = 6):
        super(ResNet, self).__init__()
        self.e_layer = ResidualBlockWithStride(N_in, N_mid, stride=2, is_filter=True)
        self.mid_layer = self._make_layer(num = num, N_mid = N_mid)
        self.d_layer = ResidualBlockUpsample(N_mid, N_mid, 2, is_filter=True)
        self.smooth = nn.Conv2d(N_mid, N_out, kernel_size=3, stride=1, padding=1)
    def _make_layer(self, num, N_mid):
        layers = list()
        for j in range(num):
            layers.append(ResidualBlock(N_mid, N_mid))
        layers.append(AttentionBlock(N_mid))
        return nn.Sequential(*layers)
    def forward(self, x):
        c = self.e_layer(x)
        c = self.mid_layer(c)
        c = self.d_layer(c)
        c = self.smooth(c)
        return c

class ResNets1(nn.Module):
    def __init__(self, N_in, N_out, N_mid=64, num = 6):
        super(ResNets1, self).__init__()
        self.enc     = nn.Conv2d(N_in, N_mid, kernel_size=3, stride=1, padding=1)
        self.e_layer  = ResidualBlockWithStride(N_mid, N_mid, stride=1, is_filter=True)
        self.mid_layer = self._make_layer(num = num, N_mid = N_mid)
        self.d_layer  = ResidualBlockUpsample(N_mid, N_mid, 1, is_filter=True)
        self.smooth   = nn.Conv2d(N_mid, N_out, kernel_size=3, stride=1, padding=1)
    def _make_layer(self, num, N_mid):
        layers = list()
        for j in range(num):
            layers.append(ResidualBlock(N_mid, N_mid))
        layers.append(AttentionBlock(N_mid))
        return nn.Sequential(*layers)
    def forward(self, x):
        c = self.enc(x)
        c = self.e_layer(c)
        c = self.mid_layer(c)
        c = self.d_layer(c)
        c = self.smooth(c)
        return c    
     
class RPLVC(CompressionModel):
    def __init__(self, N=128, M=128, Nf=128, Mf=128, **kwargs):
        super().__init__(entropy_bottleneck_channels=N, **kwargs)

        class MENet(nn.Module):
            """Motion Estimation Net"""
            def __init__(self):
                super(MENet, self).__init__()
                class MEBasic(nn.Module):
                    def __init__(self, layername):
                        super(MEBasic, self).__init__()
                        self.conv1 = nn.Conv2d(8, 32, 7, 1, padding=3)
                        self.relu1 = nn.ReLU()
                        self.conv2 = nn.Conv2d(32, 64, 7, 1, padding=3)
                        self.relu2 = nn.ReLU()
                        self.conv3 = nn.Conv2d(64, 32, 7, 1, padding=3)
                        self.relu3 = nn.ReLU()
                        self.conv4 = nn.Conv2d(32, 16, 7, 1, padding=3)
                        self.relu4 = nn.ReLU()
                        self.conv5 = nn.Conv2d(16, 2, 7, 1, padding=3)

                    def forward(self, x):
                        x = self.relu1(self.conv1(x))
                        x = self.relu2(self.conv2(x))
                        x = self.relu3(self.conv3(x))
                        x = self.relu4(self.conv4(x))
                        x = self.conv5(x)
                        return x
                    
                class ME_Spynet(nn.Module):
                    def __init__(self, layername='motion_estimation'):
                        super(ME_Spynet, self).__init__()
                        self.L = 4
                        self.moduleBasic = torch.nn.ModuleList([ MEBasic(layername + 'modelL' + str(intLevel + 1)) for intLevel in range(self.L) ])

                    def forward(self, im1, im2):
                        batchsize = im1.size()[0]
                        im1_pre = im1
                        im2_pre = im2

                        im1list = [im1_pre]
                        im2list = [im2_pre]
                        for intLevel in range(self.L - 1):
                            im1list.append(F.avg_pool2d(im1list[intLevel], kernel_size=2, stride=2))# , count_include_pad=False))
                            im2list.append(F.avg_pool2d(im2list[intLevel], kernel_size=2, stride=2))#, count_include_pad=False))

                        shape_fine = im2list[self.L - 1].size()
                        zeroshape = [batchsize, 2, shape_fine[2] // 2, shape_fine[3] // 2]
                        device_id = im1.device.index
                        flowfileds = torch.zeros(zeroshape, dtype=torch.float32, device=device_id)
                        for intLevel in range(self.L):
                            flowfiledsUpsample = bilinearupsacling(flowfileds) * 2.0
                            flowfileds = flowfiledsUpsample + self.moduleBasic[intLevel](torch.cat([im1list[self.L - 1 - intLevel], warp(im2list[self.L - 1 - intLevel], flowfiledsUpsample), flowfiledsUpsample], 1))# residualflow
                        
                        return flowfileds
                    
                def bilinearupsacling(inputfeature):
                    inputheight = inputfeature.size()[2]
                    inputwidth = inputfeature.size()[3]
                    outfeature = F.interpolate(inputfeature, (inputheight * 2, inputwidth * 2), mode='bilinear')
                    return outfeature
                self.ME_Spynet = ME_Spynet()
            def forward(self, x, ref):
                mv = self.ME_Spynet(x, ref)
                return mv
            
        class MVPNet(nn.Module):
            """Motion Vector Prediction Net"""
            def __init__(self, N_mid=64):
                super(MVPNet, self).__init__()
                self.filter = ResNet(N_in=8, N_out=2, N_mid=N_mid)#FPN(N_in=20, N_out=2, N_mid=N_mid)
                
            def forward(self, mv_hat2, mv_hat3, mv_hat4):
                mv_pred_ = warp(mv_hat2, mv_hat2)
                x = torch.cat((mv_pred_, mv_hat2, mv_hat3, mv_hat4), dim=1)
                x = self.filter(x)
                return x
                
        class RESIPNet(nn.Module):
            """Residual Prediction Net"""
            def __init__(self, N1=8, N_mid=64):
                super(RESIPNet, self).__init__()
                self.FeatureExtractor = nn.Sequential(
                    conv(3, N1, kernel_size=3, stride=1),
                    nn.LeakyReLU(inplace=True),
                )
                self.FeatureExtractor2 = nn.Sequential(
                    conv(3, N1, kernel_size=3, stride=1),
                    nn.LeakyReLU(inplace=True),
                )
                self.FeatureExtractor3 = nn.Sequential(
                    conv(3, N1, kernel_size=3, stride=1),
                    nn.LeakyReLU(inplace=True),
                )
                self.FeatureExtractor4 = nn.Sequential(
                    conv(3, N1, kernel_size=3, stride=1),
                    nn.LeakyReLU(inplace=True),
                )
                self.filter = ResNet(N_in=N1*8, N_out=3, N_mid=N_mid)#FPN(N_in=15, N_out=3, N_mid=N_mid)
                
            def forward(self, mv_hat, mv2_hat, mv3_hat, mv4_hat, resi_ref, resi_ref2, resi_ref3, resi_ref4):
                # Feature Extractor
                resi_ref_f = self.FeatureExtractor(resi_ref)
                resi_ref2_f = self.FeatureExtractor2(resi_ref2)
                resi_ref3_f = self.FeatureExtractor3(resi_ref3)
                resi_ref4_f = self.FeatureExtractor4(resi_ref4)
                
                # Feature Alignment
                resi_pred_ = warp(resi_ref_f, mv_hat.detach())
                resi_pred_2_ = warp(resi_ref2_f, mv2_hat.detach())
                resi_pred_3_ = warp(resi_ref3_f, mv3_hat.detach())
                resi_pred_4_ = warp(resi_ref4_f, mv4_hat.detach())
              
                # Concat & Filter
                x = torch.cat((resi_pred_, resi_pred_2_, resi_pred_3_, resi_pred_4_, resi_ref_f, resi_ref2_f, resi_ref3_f, resi_ref4_f), dim=1)
                resi_pred = self.filter(x)
                
                return resi_pred
            
        class MPMCNet(nn.Module):
            """Motion Compensation Net with Multiple Prediction"""
            def __init__(self, N1=8, N_mid=64):
                super(MPMCNet, self).__init__()
                self.FeatureExtractor = nn.Sequential(
                    conv(3, N1, kernel_size=3, stride=1),
                    nn.LeakyReLU(inplace=True),
                )
                self.FeatureExtractor2 = nn.Sequential(
                    conv(3, N1, kernel_size=3, stride=1),
                    nn.LeakyReLU(inplace=True),
                )
                self.FeatureExtractor3 = nn.Sequential(
                    conv(3, N1, kernel_size=3, stride=1),
                    nn.LeakyReLU(inplace=True),
                )
                self.FeatureExtractor4 = nn.Sequential(
                    conv(3, N1, kernel_size=3, stride=1),
                    nn.LeakyReLU(inplace=True),
                )
                """
                self.warp_filter = nn.Sequential(
                    conv(12, N_mid, kernel_size=3, stride=1),
                    nn.LeakyReLU(inplace=True),
                    conv(N_mid, 3, kernel_size=3, stride=1),
                )
                """
                self.filter_cat = ResNet(N_in=6+N1*8, N_out=N_mid, N_mid=N_mid)
                self.smooth = conv(N_mid, 3, kernel_size=1, stride=1)
                #self.offset = 0.1
            def forward(self, mv_hat, mv2_hat, mv3_hat, mv4_hat, ref, ref2, ref3, ref4):
                # Feature Extractor
                ref_f = self.FeatureExtractor(ref)
                ref2_f = self.FeatureExtractor2(ref2)
                ref3_f = self.FeatureExtractor3(ref3)
                ref4_f = self.FeatureExtractor4(ref4)
                
                # Feature Aligned
                """
                ref_w_ = torch.cat((warp(ref, torch.cat((mv_hat[:,0:1,]+self.offset, mv_hat[:,1:2,]+self.offset), dim=1)),
                             warp(ref, torch.cat((mv_hat[:,0:1,]+self.offset, mv_hat[:,1:2,]),         dim=1)),
                             warp(ref, torch.cat((mv_hat[:,0:1,]+self.offset, mv_hat[:,1:2,]-self.offset), dim=1)),
                             warp(ref, torch.cat((mv_hat[:,0:1,],         mv_hat[:,1:2,]+self.offset), dim=1)),
                             warp(ref, torch.cat((mv_hat[:,0:1,],         mv_hat[:,1:2,]),         dim=1)),
                             warp(ref, torch.cat((mv_hat[:,0:1,],         mv_hat[:,1:2,]-self.offset), dim=1)),
                             warp(ref, torch.cat((mv_hat[:,0:1,]-self.offset, mv_hat[:,1:2,]+self.offset), dim=1)),
                             warp(ref, torch.cat((mv_hat[:,0:1,]-self.offset, mv_hat[:,1:2,]),         dim=1)),
                             warp(ref, torch.cat((mv_hat[:,0:1,]-self.offset, mv_hat[:,1:2,]-self.offset), dim=1)),
                             warp(mv_hat, mv_hat),
                             ref), dim=1)
                """
                #ref_w_ = torch.cat((warp(ref, mv_hat*1.1), warp(ref, mv_hat), warp(ref, mv_hat*0.9), ref), dim=1)                
                
                #ref_w = self.warp_filter(ref_w_)
                
                ref_w = warp(ref, mv_hat)
                ref_fw = warp(ref_f, mv_hat)
                ref2_fw = warp(ref2_f, mv2_hat)
                ref3_fw = warp(ref3_f, mv3_hat)
                ref4_fw = warp(ref4_f, mv4_hat)

                # Concat & Filter
                #x = torch.cat((ref_w, ref, ref4_fw, ref3_fw, ref2_fw, ref_fw, ref4_f, ref3_f, ref2_f, ref_f), dim=1)
                x = torch.cat((ref_w, ref4_fw, ref3_fw, ref2_fw, ref_fw, ref, ref4_f, ref3_f, ref2_f, ref_f), dim=1)
                x_mpmc_feat = self.filter_cat(x)
                x_mpmc = self.smooth(x_mpmc_feat) + ref_w
                
                return x_mpmc, x_mpmc_feat, ref_w
            
        class LoopFilterNet(nn.Module):
            """ Loop Filtering Net"""
            def __init__(self, N1=32, N_mid=128, N=N):
                super(LoopFilterNet, self).__init__()
                self.filter = ResNet(N_in=3+N1+N, N_out=3, N_mid=N_mid)
                
            def forward(self, x_hat_, x_feat, resi_hat_):
                x_pred_feat = torch.cat((x_hat_, x_feat, resi_hat_), dim=1)
                x_hat = self.filter(x_pred_feat) + x_hat_
                return x_hat               
        
        """Residual Compression"""
        self.g_a = nn.Sequential(
            ResidualBlockWithStride(3, N, stride=2),
            ResidualBlock(N, N),
            ResidualBlockWithStride(N, N, stride=2),
            ResidualBlock(N, N),
            ResidualBlockWithStride(N, N, stride=2),
            ResidualBlock(N, N),
            conv3x3(N, N, stride=2),
        )
        self.g_s = nn.Sequential(
            ResidualBlock(N, N),
            ResidualBlockUpsample(N, N, 2),
            ResidualBlock(N, N),
            ResidualBlockUpsample(N, N, 2),
            ResidualBlock(N, N),
            ResidualBlockUpsample(N, N, 2),
            ResidualBlock(N, N),
            subpel_conv3x3(N, N, 2),
        )   
        self.h_a = nn.Sequential(
            conv3x3(N, N),
            nn.LeakyReLU(inplace=True),
            conv3x3(N, N),
            nn.LeakyReLU(inplace=True),
            conv3x3(N, N, stride=2),
            nn.LeakyReLU(inplace=True),
            conv3x3(N, N),
            nn.LeakyReLU(inplace=True),
            conv3x3(N, N, stride=2),
        )
        self.h_s = nn.Sequential(
            conv3x3(N, N),
            nn.LeakyReLU(inplace=True),
            subpel_conv3x3(N, N, 2),
            nn.LeakyReLU(inplace=True),
            conv3x3(N, N * 3 // 2),
            nn.LeakyReLU(inplace=True),
            subpel_conv3x3(N * 3 // 2, N * 3 // 2, 2),
            nn.LeakyReLU(inplace=True),
            conv3x3(N * 3 // 2, N * 2),
        )
        """
        self.entropy_parameters = nn.Sequential(
            nn.Conv2d(M * 12 // 3, M * 10 // 3, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(M * 10 // 3, M * 8 // 3, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(M * 8 // 3, M * 6 // 3, 1),
        )
        self.context_prediction = MaskedConv2d(
            M, 2 * M, kernel_size=5, padding=2, stride=1
        )
        #"""
        self.gaussian_conditional = GaussianConditional(None)
        
        """MV Compression"""
        self.g_a_mv = nn.Sequential(
            ResidualBlockWithStride(2, Nf, stride=2),
            ResidualBlock(Nf, Nf),
            ResidualBlockWithStride(Nf, Nf, stride=2),
            ResidualBlock(Nf, Nf),
            ResidualBlockWithStride(Nf, Nf, stride=2),
            ResidualBlock(Nf, Nf),
            conv3x3(Nf, Nf, stride=2),
        )

        self.g_s_mv = nn.Sequential(
            ResidualBlock(Nf, Nf),
            ResidualBlockUpsample(Nf, Nf, 2),
            ResidualBlock(Nf, Nf),
            ResidualBlockUpsample(Nf, Nf, 2),
            ResidualBlock(Nf, Nf),
            ResidualBlockUpsample(Nf, Nf, 2),
            ResidualBlock(Nf, Nf),
            subpel_conv3x3(Nf, 2, 2),
        )
        
        self.h_a_mv = nn.Sequential(
            conv3x3(Nf, Nf),
            nn.LeakyReLU(inplace=True),
            conv3x3(Nf, Nf),
            nn.LeakyReLU(inplace=True),
            conv3x3(Nf, Nf, stride=2),
            nn.LeakyReLU(inplace=True),
            conv3x3(Nf, Nf),
            nn.LeakyReLU(inplace=True),
            conv3x3(Nf, Nf, stride=2),
        )

        self.h_s_mv = nn.Sequential(
            conv3x3(Nf, Nf),
            nn.LeakyReLU(inplace=True),
            subpel_conv3x3(Nf, Nf, 2),
            nn.LeakyReLU(inplace=True),
            conv3x3(Nf, Nf * 3 // 2),
            nn.LeakyReLU(inplace=True),
            subpel_conv3x3(Nf * 3 // 2, Nf * 3 // 2, 2),
            nn.LeakyReLU(inplace=True),
            conv3x3(Nf * 3 // 2, Nf * 2),
        )
        self.entropy_bottleneck_mv = EntropyBottleneck(Nf)        
        """
        self.entropy_parameters_mv = nn.Sequential(
            nn.Conv2d(Mf * 12 // 3, Mf * 10 // 3, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(Mf * 10 // 3, Mf * 8 // 3, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(Mf * 8 // 3, Mf * 6 // 3, 1),
        )
        self.context_prediction_mv = MaskedConv2d(
            Mf, 2 * Mf, kernel_size=5, padding=2, stride=1
        )
        #"""
        self.gaussian_conditional_mv = GaussianConditional(None) 

        
        self.MENet = MENet()
        self.MVPNet = MVPNet(N_mid=32)
        self.RESIPNet = RESIPNet(N_mid=64)
        self.MPMCNet = MPMCNet(N_mid=64)
        self.LoopFilterNet = LoopFilterNet(N1=64, N_mid=128)
        self.smooth_resi = conv(N, 3, kernel_size=1, stride=1)
        

    @property
    def downsampling_factor(self) -> int:
        return 2 ** (4 + 2)
    
    def forward(self, ref, ref2, ref3, ref4, mv_hat2, mv_hat3, mv_hat4, resi_ref, resi_ref2, resi_ref3, resi_ref4, x):
        # motion estimation & prediction
        mv_ = self.MENet(x, ref)
        mv_pred = self.MVPNet(mv_hat2, mv_hat3, mv_hat4)
        #mv_pred = torch.zeros_like(ref)[:,:2]

        # motion compression
        y_mv_all = self.g_a_mv(mv_)
        y_mv_pred = self.g_a_mv(mv_pred)
        
        y_mvd = y_mv_all - y_mv_pred
        z_mvd = self.h_a_mv(y_mvd)
        z_hat_mvd, z_likelihoods_mvd = self.entropy_bottleneck_mv(z_mvd)
        params_mvd = self.h_s_mv(z_hat_mvd)
        # wo regressive
        scales_hat_mvd, means_hat_mvd = params_mvd.chunk(2, 1)
        y_hat_mvd, y_likelihoods_mvd = self.gaussian_conditional_mv(y_mvd, scales_hat_mvd, means=means_hat_mvd) 
        mv_hat = self.g_s_mv(y_hat_mvd + y_mv_pred)
        """# w regressive
        y_hat_mvd = self.gaussian_conditional_mv.quantize(
            y_mvd, "noise" if self.training else "dequantize"
        )
        ctx_params_mvd = self.context_prediction_mv(y_hat_mvd)
        gaussian_params_mvd = self.entropy_parameters_mv(
            torch.cat((params_mvd, ctx_params_mvd), dim=1)
        )
        scales_hat_mvd, means_hat_mvd = gaussian_params_mvd.chunk(2, 1)
        _, y_likelihoods_mvd = self.gaussian_conditional_mv(y_mvd, scales_hat_mvd, means=means_hat_mvd) 
        y_hat_mvd = y_hat_mvd / qstep_mv_g
        mv_hat = self.g_s_mv(y_hat_mvd + y_mv_pred)
        """
        # motion compensation
        x_mpmc, x_mpmc_feat, ref_w = self.MPMCNet(mv_hat, mv_hat2, mv_hat3, mv_hat4, ref, ref2, ref3, ref4)
        
        # resi generation
        resi_pred = self.RESIPNet(mv_hat, mv_hat2, mv_hat3, mv_hat4, resi_ref, resi_ref2, resi_ref3, resi_ref4)
        # resi difference compression
        y_resi = self.g_a(x - x_mpmc)
        y_resi_pred = self.g_a(resi_pred)
        y = y_resi - y_resi_pred
        z = self.h_a(y)
        z_hat, z_likelihoods = self.entropy_bottleneck(z)
        params = self.h_s(z_hat)
        # wo regressive
        scales_hat, means_hat = params.chunk(2, 1)
        y_hat, y_likelihoods = self.gaussian_conditional(y, scales_hat, means=means_hat)
        resi_feat = self.g_s(y_hat + y_resi_pred)       
        
        
        y_resi_pred_ = self.g_a(torch.zeros_like(ref))
        y_ = y_resi - y_resi_pred_
        z_ = self.h_a(y_)
        z_hat_, z_likelihoods_ = self.entropy_bottleneck(z_)
        params_ = self.h_s(z_hat_)
        # wo regressive
        scales_hat_, means_hat_ = params_.chunk(2, 1)
        y_hat_, y_likelihoods_ = self.gaussian_conditional(y_, scales_hat_, means=means_hat_)
        
        
        """# w regressive
        y_hat = self.gaussian_conditional.quantize(
            y, "noise" if self.training else "dequantize"
        )
        ctx_params = self.context_prediction(y_hat)
        gaussian_params = self.entropy_parameters(
            torch.cat((params, ctx_params), dim=1)
        )
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        _, y_likelihoods = self.gaussian_conditional(y, scales_hat, means=means_hat)
        y_hat = y_hat / qstep_g
        resi_feat = self.g_s(y_hat + y_resi_pred)
       """
        # reconstruction & filtering
        resi_hat = self.smooth_resi(resi_feat)
        x_hat_ = resi_hat + x_mpmc
        x_hat = self.LoopFilterNet(x_hat_, x_mpmc_feat, resi_feat)

        return {
            "x_hat": {"1":x_hat, "2":ref, "3":ref2, "4":ref3},
            "ref_w_raft": warp(ref, mv_),
            "mv_hat": {"1":mv_hat, "2":mv_hat2, "3":mv_hat3},
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
            "likelihoods_": {"y": y_likelihoods_, "z": z_likelihoods_},
            "mv_likelihoods": {"y_mv": y_likelihoods_mvd, "z_mv": z_likelihoods_mvd},
            "mv": mv_,
            "x_mpmc": x_mpmc,
            "x_pred": ref_w,
            "x_hat_":x_hat_,
            "resi_hat": x_hat - x_mpmc,
            "resi":x - x_mpmc,
            "resi_pred":resi_pred,
            "y_resi":y_resi,
            "y_resi_pred":y_resi_pred,
            "mv_all":y_mv_all,
            "mv_pred":y_mv_pred,}

    def compress(self, 
                 ref, ref2, ref3, ref4, 
                 mv_hat2, mv_hat3, mv_hat4, 
                 resi_ref, resi_ref2, resi_ref3, resi_ref4, 
                 x):   
        # motion estimation & prediction
        mv_ = self.MENet(x, ref)
        mv_pred = self.MVPNet(mv_hat2, mv_hat3, mv_hat4)
        #mv_pred = torch.zeros_like(ref)[:,:2]
        # motion compression
        y_mv_all = self.g_a_mv(mv_)
        y_mv_pred = self.g_a_mv(mv_pred)
        
        y_mvd = y_mv_all - y_mv_pred
        z_mvd = self.h_a_mv(y_mvd)
        z_mvd_strings = self.entropy_bottleneck_mv.compress(z_mvd)
        z_hat_mvd = self.entropy_bottleneck_mv.decompress(z_mvd_strings, z_mvd.size()[-2:])
        params_mvd = self.h_s_mv(z_hat_mvd)
        scales_hat_mvd, means_hat_mvd = params_mvd.chunk(2, 1)
        indexes_mvd = self.gaussian_conditional_mv.build_indexes(scales_hat_mvd)
        y_mvd_strings = self.gaussian_conditional_mv.compress(y_mvd, indexes_mvd, means=means_hat_mvd)
        y_hat_mvd = self.gaussian_conditional.decompress(y_mvd_strings, indexes_mvd, means=means_hat_mvd)
        mv_hat = self.g_s_mv(y_hat_mvd + y_mv_pred)

        # motion compensation
        x_mpmc, x_mpmc_feat, ref_w = self.MPMCNet(mv_hat, mv_hat2, mv_hat3, mv_hat4, ref, ref2, ref3, ref4)
        
        # resi generation
        resi_pred = self.RESIPNet(mv_hat, mv_hat2, mv_hat3, mv_hat4, resi_ref, resi_ref2, resi_ref3, resi_ref4)
        #resi_pred = torch.zeros_like(ref)
        #resi = x - x_mpmc - resi_pred

        # resi compression
        y_resi = self.g_a(x - x_mpmc)
        y_resi_pred = self.g_a(resi_pred)
        y = y_resi - y_resi_pred
        z = self.h_a(y)
        #z_hat, z_likelihoods = self.entropy_bottleneck(z)
        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])
        params = self.h_s(z_hat)
        
        # wo regressive
        scales_hat, means_hat = params.chunk(2, 1)
        indexes = self.gaussian_conditional.build_indexes(scales_hat)
        #y_hat, y_likelihoods = self.gaussian_conditional(y, scales_hat, means=means_hat)
        y_strings = self.gaussian_conditional.compress(y, indexes, means=means_hat)
        y_hat = self.gaussian_conditional.decompress(y_strings, indexes, means=means_hat)
        resi_feat = self.g_s(y_hat + y_resi_pred)

       
        # reconstruction & filtering
        resi_hat = self.smooth_resi(resi_feat)
        x_hat_ = resi_hat + x_mpmc
        x_hat = self.LoopFilterNet(x_hat_, x_mpmc_feat, resi_feat).clamp_(0, 1)

        return {
            "x_hat": {"1":x_hat, "2":ref, "3":ref2, "4":ref3},
            "ref_w_raft": warp(ref, mv_),
            "mv_hat": {"1":mv_hat, "2":mv_hat2, "3":mv_hat3},
            "strings": [y_strings, z_strings, y_mvd_strings, z_mvd_strings], 
            "shape": z.size()[-2:],
            "shape_mv": z_mvd.size()[-2:],
            "mv_raft": mv_,
            "x_mpmc": x_mpmc,
            "x_pred": ref_w,
            "x_hat_":x_hat_,
            "resi_hat": x_hat - x_mpmc,
            "resi":x - x_mpmc,
            "resi_pred":resi_pred,
            "y_resi":y_resi,
            "y_resi_pred":y_resi_pred,
            "mv_all":y_mv_all,
            "mv_pred":y_mv_pred,}
    
    def decompress(self, 
                 ref, ref2, ref3, ref4, 
                 mv_hat2, mv_hat3, mv_hat4, 
                 resi_ref, resi_ref2, resi_ref3, resi_ref4, 
                 y_strings, z_strings, y_mvd_strings, z_mvd_strings,
                 z_shape, z_mvd_shape):
        
        # motion estimation & prediction
        mv_pred = self.MVPNet(mv_hat2, mv_hat3, mv_hat4)

        # motion compression
        y_mv_pred = self.g_a_mv(mv_pred)
        
        z_hat_mvd = self.entropy_bottleneck_mv.decompress(z_mvd_strings, z_mvd_shape)
        
        params_mvd = self.h_s_mv(z_hat_mvd)
        scales_hat_mvd, means_hat_mvd = params_mvd.chunk(2, 1)
        indexes_mvd = self.gaussian_conditional_mv.build_indexes(scales_hat_mvd)
        y_hat_mvd = self.gaussian_conditional_mv.decompress(y_mvd_strings, indexes_mvd, means=means_hat_mvd)
        mv_hat = self.g_s_mv(y_hat_mvd + y_mv_pred)

        # motion compensation
        x_mpmc, x_mpmc_feat, ref_w = self.MPMCNet(mv_hat, mv_hat2, mv_hat3, mv_hat4, ref, ref2, ref3, ref4)
        
        # resi generation
        resi_pred = self.RESIPNet(mv_hat, mv_hat2, mv_hat3, mv_hat4, resi_ref, resi_ref2, resi_ref3, resi_ref4)
        y_resi_pred = self.g_a(resi_pred)
        #resi = x - x_mpmc - resi_pred

        # resi decompression
        z_hat = self.entropy_bottleneck.decompress(z_strings, z_shape)
        params = self.h_s(z_hat)
        
        # wo regressive
        scales_hat, means_hat = params.chunk(2, 1)
        indexes = self.gaussian_conditional.build_indexes(scales_hat)
        y_hat = self.gaussian_conditional.decompress(y_strings, indexes, means=means_hat)
        resi_feat = self.g_s(y_hat + y_resi_pred)

        # reconstruction & filtering
        resi_hat = self.smooth_resi(resi_feat)
        x_hat_ = resi_hat + x_mpmc
        x_hat = self.LoopFilterNet(x_hat_, x_mpmc_feat, resi_feat).clamp_(0, 1)

        return {
            "x_hat": {"1":x_hat, "2":ref, "3":ref2, "4":ref3},
            "mv_hat": {"1":mv_hat, "2":mv_hat2, "3":mv_hat3},
            "x_mpmc": x_mpmc,
            "x_pred": ref_w,
            "x_hat_":x_hat_,
            "resi_hat": x_hat - x_mpmc,
            "resi_pred":y_resi_pred,
            "mv_pred":y_mv_pred,}

    def update(self, scale_table=None, force=False):
        if scale_table is None:
            scale_table = get_scale_table()
        self.gaussian_conditional.update_scale_table(scale_table, force=force)
        self.gaussian_conditional_mv.update_scale_table(scale_table, force=force)
        super().update(force=force)

    def load_state_dict(self, state_dict):
        # Dynamically update the entropy bottleneck buffers related to the CDFs
        update_registered_buffers(
            self.entropy_bottleneck,
            "entropy_bottleneck",
            ["_quantized_cdf", "_offset", "_cdf_length"],
            state_dict,
        )
        update_registered_buffers(
            self.gaussian_conditional,
            "gaussian_conditional",
            ["_quantized_cdf", "_offset", "_cdf_length", "scale_table"],
            state_dict,
        )
        update_registered_buffers(
            self.entropy_bottleneck_mv,
            "entropy_bottleneck_mv",
            ["_quantized_cdf", "_offset", "_cdf_length"],
            state_dict,
        )
        update_registered_buffers(
            self.gaussian_conditional_mv,
            "gaussian_conditional_mv",
            ["_quantized_cdf", "_offset", "_cdf_length", "scale_table"],
            state_dict,
        )
        super().load_state_dict(state_dict)
