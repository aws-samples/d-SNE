#!/usr/bin/env bash
cd datasets

# Download the MNIST
if [ ! -d "Office31" ]; then
    mkdir "Office31"
fi

wget https://s3-us-west-2.amazonaws.com/domain-adaptation-exps/datasets/Office31/Office31.zip -O Office31/Office31.zip
unzip Office31/Office31.zip
