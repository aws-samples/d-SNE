#!/usr/bin/env bash
cd datasets

# Download the MNIST
if [ ! -d "Office31" ]; then
    mkdir "Office31"
fi

wget --no-check-certificate -O tmp.tar.gz "https://drive.google.com/uc?export=download&id=0B4IapRTv9pJ1WGZVd1VDMmhwdlE"
tar -xvzf tmp.tar.gz -C Office31
rm tmp.tar.gz