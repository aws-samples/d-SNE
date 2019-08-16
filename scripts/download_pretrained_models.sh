#!/usr/bin/env bash
# Download the pretrained model trained on the ImageNet dataset
if [ ! -d "pretrained" ]; then
    mkdir "pretrained"
fi

wget https://domain-adaptation-exps.s3-us-west-2.amazonaws.com/models/pretrained/vgg16.params -O pretrained/vgg16.params
wget https://domain-adaptation-exps.s3-us-west-2.amazonaws.com/models/pretrained/resnet101_v2.params -O pretrained/resnet101_v2.params
wget https://domain-adaptation-exps.s3-us-west-2.amazonaws.com/models -O /pretrained/resnet152_v2.params 
wget https://domain-adaptation-exps.s3-us-west-2.amazonaws.com/models/pretrained/densenet169.params -O pretrained/densenet169.params
wget https://domain-adaptation-exps.s3-us-west-2.amazonaws.com/models/pretrained/densenet201.params -O pretrained/densenet201.params 
