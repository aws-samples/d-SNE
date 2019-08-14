#!/usr/bin/env bash

cd datasets

# Download the MNIST
if [ ! -d "MNIST" ]; then
    mkdir "MNIST"
fi

wget https://s3-us-west-2.amazonaws.com/domain-adaptation-exps/datasets/MNIST/mnist.pkl -O MNIST/mnist.pkl

# Download the MNIST-M
if [ ! -d "MNIST-M" ]; then
    mkdir "MNIST-M"
fi

wget https://s3-us-west-2.amazonaws.com/domain-adaptation-exps/datasets/MNIST-M/mnist_m.pkl -O MNIST-M/mnist_m.pkl

# Download the SVHN
if [ ! -d "SVHN" ]; then
    mkdir "SVHN"
fi

wget https://s3-us-west-2.amazonaws.com/domain-adaptation-exps/datasets/SVHN/svhn.pkl -O SVHN/svhn.pkl

# Download the Digits
if [ ! -d "USPS" ]; then
    mkdir "USPS"
fi

wget https://s3-us-west-2.amazonaws.com/domain-adaptation-exps/datasets/USPS/usps.pkl -O USPS/usps.pkl