#------------------------------------------------------------------------------
    #
    #  The confidential and proprietary information contained in this file may
    #  only be used by a person authorised under and to the extent permitted
    #  by a subsisting licensing agreement from VIP Lab.
    #
    #                   (C) COPYRIGHT 2021 VIP Lab.
    #                       ALL RIGHTS RESERVED
    #
    #  This entire notice must be reproduced on all copies of this file
    #  and copies of this file may only be made by a person if such person is
    #  permitted to do so under the terms of a subsisting license agreement
    #  from VIP Lab
    #
#------------------------------------------------------------------------------
    #
    #  Filename       : test.py
    #  Author         : Liu Chao
    #  Created        : 2021-05-24
    #  Description    : Creat encode, decode, and check funcs.
    #
#------------------------------------------------------------------------------

import os
import time
import struct
import sys
import argparse
import math
import numpy   as np
from   numpy   import mean
from   pathlib import Path
from   PIL     import Image

sys.path.append(".")

import torch
import compressai
from   tools          import *
from   pytorch_msssim import ms_ssim
from   torchvision    import transforms


def ceil_base(x, base = 2):
    return int(np.ceil((x)/base))*base

def compress_video(net,
                   video,
                   video_root_path,
                   coded_frame_num,
                   gop,
                   bit_path,
                   rec_path,
                   bpg_coding,
                   QP,
                   vtm_coding,
                   verbose):
    enc_start = time.time()
    print("Encoding "+video)
    device = next(net.parameters()).device
    psnr_list     = list()
    msssim_list   = list()
    bpp_resi_list = list()
    bpp_mv_list   = list()
    bpp_all_list  = list()
    with torch.no_grad():        
        net.update()
        for i in range(1, coded_frame_num + 1):
            t_start = time.time()
            x_path = os.path.join(video_root_path, video, "img"+str(i).zfill(6)+".png")
            x = img2torch(Image.open(x_path)).to(device)
            h_y, w_y = x.shape[-2], x.shape[-1]
            if i % gop == 1:
                if bpg_coding:
                    source_img_path = x_path
                    coded_ibin_path = os.path.join(bit_path, '_'.join([video,str(i).zfill(6),'bpg.bin']))
                    rec_img_path    = os.path.join(rec_path, '_'.join([video,str(i).zfill(6),'bpg.png']))
                    cmd = ' '.join(['./bin/bpgenc','-f','444','-m','9',source_img_path,'-o',coded_ibin_path,'-q',str(QP)])
                    os.system(cmd)
                    cmd = ' '.join(["./bin/bpgdec",coded_ibin_path,"-o",rec_img_path])
                    os.system(cmd)
                    x_hat = img2torch(Image.open(rec_img_path)).to(device)
                elif vtm_coding:
                    pass
                ref       = x_hat
                ref2      = ref
                ref3      = ref2
                ref4      = ref3
                mv_hat2    = torch.zeros_like(ref)[:,:2]
                mv_hat3    = mv_hat2
                mv_hat4    = mv_hat3
                resi_ref   = torch.zeros_like(ref)
                resi_ref2   = resi_ref
                resi_ref3   = resi_ref2
                resi_ref4   = resi_ref3
                bpp_resi   = 0
                bpp_mv     = 0
                bpp_all    = float(os.path.getsize(coded_ibin_path)) * 8 / (h_y * w_y)

            else:
                out_hat   = net.compress(ref, ref2, ref3, ref4, 
                                         mv_hat2, mv_hat3, mv_hat4, 
                                         resi_ref, resi_ref2, resi_ref3, resi_ref4, 
                                         x)
                x_hat     = out_hat['x_hat']['1']
                ref4      = ref3
                ref3      = ref2
                ref2      = ref
                ref       = out_hat['x_hat']['1'].detach()
                mv_hat4    = mv_hat3
                mv_hat3    = mv_hat2
                mv_hat2    = out_hat['mv_hat']['1'].detach()
                resi_ref4  = resi_ref3
                resi_ref3  = resi_ref2
                resi_ref2  = resi_ref
                resi_ref   = out_hat['resi_hat'].detach()
                rateList   = []
                coded_pbin_path = os.path.join(bit_path, '_'.join([video,'lvc.bin']))
                with Path(coded_pbin_path).open("ab") as f:
                    for s in out_hat["strings"]:
                        write_uints(f, (len(s[0]),))
                        write_bytes(f, s[0])
                        rateList.append(len(s[0]))
                bpp_resi = (sum(rateList[:2]) + 8) * 8 / (h_y * w_y)
                bpp_mv   = (sum(rateList[2:]) + 8) * 8 / (h_y * w_y)
                bpp_resi_list.append(bpp_resi)
                bpp_mv_list.append(bpp_mv)
                bpp_all  = bpp_resi + bpp_mv
            psnr     = compute_psnr(x, x_hat)
            msssim   = compute_msssim(x, x_hat)
            psnr_list.append(psnr)
            msssim_list.append(msssim)
            bpp_all_list.append(bpp_all)
            x_hat = Image.fromarray((np.array(x_hat.cpu())[0]*255).transpose(1,2,0).astype('uint8'))
            x_hat.save(os.path.join(rec_path, '_'.join([video,str(i).zfill(6)+'.png'])))
            enc_time = time.time() - t_start

            if verbose >= 2:
                print(f" POC {i} |"
                      f" Time {enc_time:.2f}s, hat mode |"
                      f" psnr {psnr:.4f} |"
                      f" ms-ssim {msssim:.4f} |"
                      f" bpp_all {bpp_all:.4f} |"
                      f" bpp_resi {bpp_resi:.4f} |"
                      f" bpp_mv {bpp_mv:.4f}"
                )
    
    enc_time = time.time() - enc_start
    if verbose >= 1:
        print(f" Encoded in {enc_time:.2f}s, hat mode |"
              f" psnr {mean(psnr_list):.3f} |"
              f" ms-ssim {mean(msssim_list):.5f} |"
              f" bpp {mean(bpp_all_list):.4f}"
             )
    return psnr_list, msssim_list, bpp_resi_list, bpp_mv_list, bpp_all_list

def decompress_video(net,
                       coded_frame_num,
                       gop,
                       bit,
                       bit_path,
                       rec_path,
                       bpg_decoding,
                       vtm_decoding,
                       verbose):
    enc_start = time.time()
    print("Decoding "+bit)
    device = next(net.parameters()).device
    n_strings = 4
    video = bit.split('_')[0]
    with torch.no_grad():
        net.update()
        with Path(os.path.join(bit_path, bit)).open("rb") as f:      
            for i in range(1, coded_frame_num + 1):
                t_start = time.time()
                if i % gop == 1:
                    if bpg_decoding:
                        coded_ibin_path = os.path.join(bit_path, '_'.join([video,str(i).zfill(6),'bpg.bin']))
                        rec_img_path    = os.path.join(rec_path, '_'.join([video,str(i).zfill(6),'bpg.png']))
                        cmd = ' '.join(["./bin/bpgdec",coded_ibin_path,"-o",rec_img_path])
                        os.system(cmd)
                        x_hat = img2torch(Image.open(rec_img_path)).to(device)
                        h_y, w_y = x_hat.shape[-2], x_hat.shape[-1]
                        base = 64
                        z_shape = (ceil_base(h_y, base = base) // base, ceil_base(w_y, base = base) // base)
                        z_mvd_shape = z_shape
                    elif vtm_decoding:
                        pass
                    ref       = x_hat
                    ref2      = ref
                    ref3      = ref2
                    ref4      = ref3
                    mv_hat2   = torch.zeros_like(ref)[:,:2]
                    mv_hat3   = mv_hat2
                    mv_hat4   = mv_hat3
                    resi_ref  = torch.zeros_like(ref)
                    resi_ref2 = resi_ref
                    resi_ref3 = resi_ref2
                    resi_ref4 = resi_ref3
                    bpp_resi  = 0
                    bpp_mv    = 0
                    bpp_all   = float(os.path.getsize(coded_ibin_path)) * 8 / (h_y * w_y)

                else:
                    strings = []
                    for _ in range(n_strings):
                        s = read_bytes(f, read_uints(f, 1)[0])
                        strings.append([s])
                    y_strings, z_strings, y_mvd_strings, z_mvd_strings = strings
                    bpp_resi  = (len(y_strings[0]) + len(z_strings[0]) + 8) * 8 / (h_y * w_y)
                    bpp_mv    = (len(y_mvd_strings[0]) + len(z_mvd_strings[0]) + 8) * 8 / (h_y * w_y)
                    bpp_all   = bpp_resi + bpp_mv
                    out_hat   = net.decompress(ref, ref2, ref3, ref4, 
                                               mv_hat2, mv_hat3, mv_hat4, 
                                               resi_ref, resi_ref2, resi_ref3, resi_ref4, 
                                               y_strings, z_strings, y_mvd_strings, z_mvd_strings,
                                               z_shape, z_mvd_shape)

                    x_hat     = out_hat['x_hat']['1']
                    ref4      = ref3
                    ref3      = ref2
                    ref2      = ref
                    ref       = out_hat['x_hat']['1'].detach()
                    mv_hat4   = mv_hat3
                    mv_hat3   = mv_hat2
                    mv_hat2   = out_hat['mv_hat']['1'].detach()
                    resi_ref4 = resi_ref3
                    resi_ref3 = resi_ref2
                    resi_ref2 = resi_ref
                    resi_ref  = out_hat['resi_hat'].detach()

                x_hat = Image.fromarray((np.array(x_hat.cpu())[0]*255).transpose(1,2,0).astype('uint8'))
                x_hat.save(os.path.join(rec_path, '_'.join([video,str(i).zfill(6)+'.png'])))
                dec_time = time.time() - t_start

                if verbose >= 2:
                    print(f" POC {i} |"
                          f" bpp {bpp_all:.4f} |"
                          f" Time {dec_time:.2f}s, hat mode |"
                    )
    dec_time = time.time() - enc_start
    if verbose >= 1:
        print(f" Decoded in {dec_time:.2f}s, hat mode"
        )

def parse_args(argv):
    parser = argparse.ArgumentParser(description="Test script")
    parser.add_argument(
        '-g',
        '--gpu',
        type=str,
        default='0',
        help='The gpu index for testing. [0]')
    parser.add_argument(
        '-m',
        '--model',
        default='0',
        type=str,
        help='Models index. [0]')
    parser.add_argument(
        '-q',
        '--qp',
        type=int,
        default=22,
        help='QP for BPG reconstruction. [22]')
    parser.add_argument(
        '--gop',
        type=int,
        default=100,
        help='Group of Pictures. [100]')
    parser.add_argument(
        '-v',
        '--verbose',
        default=1,
        type=int,
        help='log level.[1] 0 - Class-level log; 1 - Video-level log; 2 - Frame-level log.')
    parser.add_argument(
        '-e',
        '--encode',
        default=False,
        type=bool,
        help='Start encoding videos.')
    parser.add_argument(
        '-d',
        '--decode',
        default=False,
        type=bool,
        help='Start decoding videos.')
    parser.add_argument(
        '-c',
        '--check',
        default=False,
        type=bool,
        help='Start check videos.')
    args = parser.parse_args(argv)
    return args

def test(argv):
    args = parse_args(argv)
    os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu    
    
    # set entropy coder
    coder = "ans"
    compressai.set_entropy_coder(coder)
    
    # load models
    from compressai.models import RPLVC
    net = RPLVC(N=128, Nf=128).cpu().eval()
    if torch.cuda.is_available():
        net = net.cuda()
    PATH = "./pretrained/checkpoint" + args.model + ".pth.tar"
    QP = args.qp
    gop = args.gop
    net.load_state_dict(torch.load(PATH, map_location='cpu')['state_dict'])    

    #for dataset in ['ClassD','ClassC','ClassE','ClassB']:
    for dataset in ['ClassD']:
        video_root_path = os.path.join("./datasets/", dataset)
        bit_path        = os.path.join('./bits/', PATH.split('/')[-1].split('.')[0], dataset)
        rec_path        = os.path.join('./recs/', PATH.split('/')[-1].split('.')[0], dataset)
        rec_dec_path    = os.path.join('./recs_dec/', PATH.split('/')[-1].split('.')[0], dataset)
        if args.encode:
            t_start = time.time()
            os.system(" ".join(["mkdir", "-p", bit_path, rec_path, rec_dec_path]))
            #os.system(" ".join(["rm", bit_path+"/*", rec_path+"/*"]))
            psnr_all_list   = []
            msssim_all_list = []
            bpp_all_list    = []
            for video in os.listdir(video_root_path):
                coded_frame_num = 100
                psnr_list, msssim_list, bpp_resi_list, bpp_mv_list, bpp_sum_list = compress_video(net,
                                                                                                  video,
                                                                                                  video_root_path,
                                                                                                  coded_frame_num,
                                                                                                  gop,
                                                                                                  bit_path,
                                                                                                  rec_path,
                                                                                                  bpg_coding = True,
                                                                                                  QP         = QP,
                                                                                                  vtm_coding = False,
                                                                                                  verbose    = args.verbose)
                assert len(psnr_list) == coded_frame_num
                psnr_all_list.append(mean(psnr_list))
                msssim_all_list.append(mean(msssim_list))
                bpp_all_list.append(mean(bpp_sum_list))
            enc_time = time.time() - t_start
            print(dataset)
            print(f" Encoded in {enc_time:.2f}s, hat mode |"
                  f" psnr {mean(psnr_all_list):.4f} |"
                  f" ms-ssim {mean(msssim_all_list):.4f} |"
                  f" bpp {mean(bpp_all_list):.4f}\n"
            )
        if args.decode:
            t_start = time.time()
            #os.system(" ".join(["rm", rec_dec_path+"/*"]))
            for bit in os.listdir(bit_path):
                if "lvc.bin" in bit:
                    coded_frame_num = 100
                    decompress_video(net,
                                     coded_frame_num,
                                     gop,
                                     bit,
                                     bit_path,
                                     rec_dec_path,
                                     bpg_decoding = True,
                                     vtm_decoding = False,
                                     verbose = args.verbose)
            dec_time = time.time() - t_start
            print("Summary:")
            print(f" Decoded in {dec_time:.2f}s, hat mode\n")
        if args.check:
            print("Checking " + dataset)
            for recs in os.listdir(rec_path):
                enc_rec = os.path.join(rec_path,     recs)
                dec_rec = os.path.join(rec_dec_path, recs)
                assert os.system(" ".join(["cmp", enc_rec, dec_rec])) == 0, "MISMATCH!!!"
            print("  Check Pass!")

if __name__ == "__main__":
    test(sys.argv[1:])