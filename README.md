
## Environment settings
Docker is used to build the environment to ensure consistent test results.

Pull pytorch-1.6 docker image.

    docker pull pytorch/pytorch:1.6.0-cuda10.1-cudnn7-runtime

Create docker environment

    sudo nvidia-docker run -v $(pwd):$(pwd) -v /etc/localtime:/etc/localtime:ro -it -w $(pwd) pytorch/pytorch:1.6.0-cuda10.1-cudnn7-runtime

Install some necessary dependencies

    pip install scipy

## Download test sets and pre-trained models

The downloaded pretrained models and the input videos should be placed in the pretrained and datasets folder. Here we have given the models and the HEVC Class-C Datasets on Google Cloud Drive.

We have extended xx and xx to crop the data. The crop size is carried over from xx, using 64.

    ffmpeg -s WxH -pix_fmt yuv420p -i input_video.yuv -vframes N -vf crop=W_:H_:0:0 -f image2 output_path/img%06d.png
where W,H denotes the width and height of the input sequence. Notation W_, H_ denotes the width and height of the output one, and N denotes the number of frames to be coded.

## Testing
The testing process consists of two main steps: encoding and decoding. In order to ensure the correctness of coding, it is also important to check the matching of encoding and decoding results. Their respective commands are as follows:

Encoding

    python test-rgb.py --model 0 --qp 27 --verbose 1 --gpu 0 --encode 1

Decoding

    python test-rgb.py --model 0 --qp 27 --verbose 1 --gpu 0 --decode 1

Mismatch Check

    python test-rgb.py --check 1

All the three can also be executed at once

    python test-rgb.py --model 0 --qp 27 --verbose 1 --gpu 0 --encode 1 --decode 1 --check 1

Parameter Descriptions
Param. |Type|Default| Descriptions
:-:|:-:|:-:|:-
model |Int|0| Index of the model, the larger the index the higher the reconstructed quality. Range[0, 4]
qp |Int|22|QP value of BPG used for the 1-st I-frame coding.
verbose |Int|1|log level. 0 - Class-level log; 1 - Video-level log; 2 - Frame-level log.
gpu |Int|0|Index of the GPU for testing.
encode |Bool|False|Start encoding Flag.
decode |Bool|False|Start decoding Flag.
check |Bool|False|Start checking Flag.


The bitrate allocation between I, P frames can be flexible, and here we give some recommended QP values of BPG as shown in follow.

Model Idx| BPG QP
:-:|:-:
0 |27
1 |27
2 |27
3 |22
4 |22