#!/usr/bin/env bash

python train_office.py --method v1 --src A --tgt D --bb vgg --nlayers 16 --model-path SRC-TUNE/vgg16.params --hybridize --end-epoch 20 --postfix t0 --flip --cfg cfg/office31s.json --log-itv 10000
python train_office.py --method v1 --src A --tgt W --bb vgg --nlayers 16 --model-path SRC-TUNE/vgg16.params --hybridize --end-epoch 20 --postfix t0 --flip --cfg cfg/office31s.json --log-itv 10000
python train_office.py --method v1 --src D --tgt A --bb vgg --nlayers 16 --model-path SRC-TUNE/vgg16.params --hybridize --end-epoch 20 --postfix t0 --flip --cfg cfg/office31s.json --log-itv 10000
python train_office.py --method v1 --src D --tgt W --bb vgg --nlayers 16 --model-path SRC-TUNE/vgg16.params --hybridize --end-epoch 20 --postfix t0 --flip --cfg cfg/office31s.json --log-itv 10000
python train_office.py --method v1 --src W --tgt A --bb vgg --nlayers 16 --model-path SRC-TUNE/vgg16.params --hybridize --end-epoch 20 --postfix t0 --flip --cfg cfg/office31s.json --log-itv 10000
python train_office.py --method v1 --src W --tgt D --bb vgg --nlayers 16 --model-path SRC-TUNE/vgg16.params --hybridize --end-epoch 20 --postfix t0 --flip --cfg cfg/office31s.json --log-itv 10000

python train_office.py --method v1 --src A --tgt D --bb vgg --nlayers 16 --model-path SRC-TUNE/vgg16.params --hybridize --end-epoch 20 --postfix t1 --flip --cfg cfg/office31s.json --log-itv 10000
python train_office.py --method v1 --src A --tgt W --bb vgg --nlayers 16 --model-path SRC-TUNE/vgg16.params --hybridize --end-epoch 20 --postfix t1 --flip --cfg cfg/office31s.json --log-itv 10000
python train_office.py --method v1 --src D --tgt A --bb vgg --nlayers 16 --model-path SRC-TUNE/vgg16.params --hybridize --end-epoch 20 --postfix t1 --flip --cfg cfg/office31s.json --log-itv 10000
python train_office.py --method v1 --src D --tgt W --bb vgg --nlayers 16 --model-path SRC-TUNE/vgg16.params --hybridize --end-epoch 20 --postfix t1 --flip --cfg cfg/office31s.json --log-itv 10000
python train_office.py --method v1 --src W --tgt A --bb vgg --nlayers 16 --model-path SRC-TUNE/vgg16.params --hybridize --end-epoch 20 --postfix t1 --flip --cfg cfg/office31s.json --log-itv 10000
python train_office.py --method v1 --src W --tgt D --bb vgg --nlayers 16 --model-path SRC-TUNE/vgg16.params --hybridize --end-epoch 20 --postfix t1 --flip --cfg cfg/office31s.json --log-itv 10000

python train_office.py --method v1 --src A --tgt D --bb vgg --nlayers 16 --model-path SRC-TUNE/vgg16.params --hybridize --end-epoch 20 --postfix t2 --flip --cfg cfg/office31s.json --log-itv 10000
python train_office.py --method v1 --src A --tgt W --bb vgg --nlayers 16 --model-path SRC-TUNE/vgg16.params --hybridize --end-epoch 20 --postfix t2 --flip --cfg cfg/office31s.json --log-itv 10000
python train_office.py --method v1 --src D --tgt A --bb vgg --nlayers 16 --model-path SRC-TUNE/vgg16.params --hybridize --end-epoch 20 --postfix t2 --flip --cfg cfg/office31s.json --log-itv 10000
python train_office.py --method v1 --src D --tgt W --bb vgg --nlayers 16 --model-path SRC-TUNE/vgg16.params --hybridize --end-epoch 20 --postfix t2 --flip --cfg cfg/office31s.json --log-itv 10000
python train_office.py --method v1 --src W --tgt A --bb vgg --nlayers 16 --model-path SRC-TUNE/vgg16.params --hybridize --end-epoch 20 --postfix t2 --flip --cfg cfg/office31s.json --log-itv 10000
python train_office.py --method v1 --src W --tgt D --bb vgg --nlayers 16 --model-path SRC-TUNE/vgg16.params --hybridize --end-epoch 20 --postfix t2 --flip --cfg cfg/office31s.json --log-itv 10000
