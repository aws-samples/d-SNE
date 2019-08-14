#!/usr/bin/env bash
# This scripts is used to tune the pretrained model for office 31 experiments
python train_office.py --method v0 --src A --tgt D --bb vgg --nlayers 16 --pretrained --model-path pretrained/vgg16.params --hybridize --train-src --end-epoch 50 --postfix p0 --flip --random-crop --random-color --cfg cfg/office31t.json
python train_office.py --method v0 --src A --tgt W --bb vgg --nlayers 16 --pretrained --model-path pretrained/vgg16.params --hybridize --train-src --end-epoch 50 --postfix p0 --flip --random-crop --random-color --cfg cfg/office31t.json
python train_office.py --method v0 --src D --tgt A --bb vgg --nlayers 16 --pretrained --model-path pretrained/vgg16.params --hybridize --train-src --end-epoch 50 --postfix p0 --flip --random-crop --random-color --cfg cfg/office31t.json
python train_office.py --method v0 --src D --tgt W --bb vgg --nlayers 16 --pretrained --model-path pretrained/vgg16.params --hybridize --train-src --end-epoch 50 --postfix p0 --flip --random-crop --random-color --cfg cfg/office31t.json
python train_office.py --method v0 --src W --tgt A --bb vgg --nlayers 16 --pretrained --model-path pretrained/vgg16.params --hybridize --train-src --end-epoch 50 --postfix p0 --flip --random-crop --random-color --cfg cfg/office31t.json
python train_office.py --method v0 --src W --tgt D --bb vgg --nlayers 16 --pretrained --model-path pretrained/vgg16.params --hybridize --train-src --end-epoch 50 --postfix p0 --flip --random-crop --random-color --cfg cfg/office31t.json

#python office.py --method v0 --src A --tgt D --bb vgg --nlayers 16 --pretrained --model-path pretrained/vgg16.params --hybridize --train-src --end-epoch 20 --postfix p1 --flip --cfg cfg/office31-t.json --log-itv 10000
#python office.py --method v0 --src A --tgt W --bb vgg --nlayers 16 --pretrained --model-path pretrained/vgg16.params --hybridize --train-src --end-epoch 20 --postfix p1 --flip --cfg cfg/office31-t.json --log-itv 10000
#python office.py --method v0 --src D --tgt A --bb vgg --nlayers 16 --pretrained --model-path pretrained/vgg16.params --hybridize --train-src --end-epoch 20 --postfix p1 --flip --cfg cfg/office31-t.json --log-itv 10000
#python office.py --method v0 --src D --tgt W --bb vgg --nlayers 16 --pretrained --model-path pretrained/vgg16.params --hybridize --train-src --end-epoch 20 --postfix p1 --flip --cfg cfg/office31-t.json --log-itv 10000
#python office.py --method v0 --src W --tgt A --bb vgg --nlayers 16 --pretrained --model-path pretrained/vgg16.params --hybridize --train-src --end-epoch 20 --postfix p1 --flip --cfg cfg/office31-t.json --log-itv 10000
#python office.py --method v0 --src W --tgt D --bb vgg --nlayers 16 --pretrained --model-path pretrained/vgg16.params --hybridize --train-src --end-epoch 20 --postfix p1 --flip --cfg cfg/office31-t.json --log-itv 10000


python train_office.py --method v0 --src A --tgt D --bb vgg --nlayers 16 --pretrained --model-path pretrained/vgg16.params --hybridize --train-src --end-epoch 50 --postfix p0-l2n --flip --cfg cfg/office31t.json --log-itv 10000 --l2n --lr 0.01
python train_office.py --method v0 --src A --tgt W --bb vgg --nlayers 16 --pretrained --model-path pretrained/vgg16.params --hybridize --train-src --end-epoch 50 --postfix p0-l2n --flip --cfg cfg/office31t.json --log-itv 10000 --l2n --lr 0.01
python train_office.py --method v0 --src D --tgt A --bb vgg --nlayers 16 --pretrained --model-path pretrained/vgg16.params --hybridize --train-src --end-epoch 50 --postfix p0-l2n --flip --cfg cfg/office31t.json --log-itv 10000 --l2n --lr 0.01
python train_office.py --method v0 --src D --tgt W --bb vgg --nlayers 16 --pretrained --model-path pretrained/vgg16.params --hybridize --train-src --end-epoch 50 --postfix p0-l2n --flip --cfg cfg/office31t.json --log-itv 10000 --l2n --lr 0.01
python train_office.py --method v0 --src W --tgt A --bb vgg --nlayers 16 --pretrained --model-path pretrained/vgg16.params --hybridize --train-src --end-epoch 50 --postfix p0-l2n --flip --cfg cfg/office31t.json --log-itv 10000 --l2n --lr 0.01
python train_office.py --method v0 --src W --tgt D --bb vgg --nlayers 16 --pretrained --model-path pretrained/vgg16.params --hybridize --train-src --end-epoch 50 --postfix p0-l2n --flip --cfg cfg/office31t.json --log-itv 10000 --l2n --lr 0.01


# Repeat 2
python train_office.py --method v0 --src A --tgt D --bb resnet --nlayers 101 --pretrained --model-path pretrained/resnet101_v2.params --hybridize --train-src --end-epoch 50 --postfix p1 --flip --cfg cfg/office31t.json --log-itv 10000
python train_office.py --method v0 --src A --tgt W --bb resnet --nlayers 101 --pretrained --model-path pretrained/resnet101_v2.params --hybridize --train-src --end-epoch 50 --postfix p1 --flip --cfg cfg/office31t.json --log-itv 10000
python train_office.py --method v0 --src D --tgt A --bb resnet --nlayers 101 --pretrained --model-path pretrained/resnet101_v2.params --hybridize --train-src --end-epoch 50 --postfix p1 --flip --cfg cfg/office31t.json --log-itv 10000
python train_office.py --method v0 --src D --tgt W --bb resnet --nlayers 101 --pretrained --model-path pretrained/resnet101_v2.params --hybridize --train-src --end-epoch 50 --postfix p1 --flip --cfg cfg/office31t.json --log-itv 10000
python train_office.py --method v0 --src W --tgt A --bb resnet --nlayers 101 --pretrained --model-path pretrained/resnet101_v2.params --hybridize --train-src --end-epoch 50 --postfix p1 --flip --cfg cfg/office31t.json --log-itv 10000
python train_office.py --method v0 --src W --tgt D --bb resnet --nlayers 101 --pretrained --model-path pretrained/resnet101_v2.params --hybridize --train-src --end-epoch 50 --postfix p1 --flip --cfg cfg/office31t.json --log-itv 10000

python train_office.py --method v0 --src A --tgt D --bb vgg --nlayers 16 --pretrained --model-path pretrained/vgg16.params --hybridize --train-src --end-epoch 50 --postfix p1-l2n --flip --cfg cfg/office31t.json --log-itv 10000 --l2n --lr 0.01
python train_office.py --method v0 --src A --tgt W --bb vgg --nlayers 16 --pretrained --model-path pretrained/vgg16.params --hybridize --train-src --end-epoch 50 --postfix p1-l2n --flip --cfg cfg/office31t.json --log-itv 10000 --l2n --lr 0.01
python train_office.py --method v0 --src D --tgt A --bb vgg --nlayers 16 --pretrained --model-path pretrained/vgg16.params --hybridize --train-src --end-epoch 50 --postfix p1-l2n --flip --cfg cfg/office31t.json --log-itv 10000 --l2n --lr 0.01
python train_office.py --method v0 --src D --tgt W --bb vgg --nlayers 16 --pretrained --model-path pretrained/vgg16.params --hybridize --train-src --end-epoch 50 --postfix p1-l2n --flip --cfg cfg/office31t.json --log-itv 10000 --l2n --lr 0.01
python train_office.py --method v0 --src W --tgt A --bb vgg --nlayers 16 --pretrained --model-path pretrained/vgg16.params --hybridize --train-src --end-epoch 50 --postfix p1-l2n --flip --cfg cfg/office31t.json --log-itv 10000 --l2n --lr 0.01
python train_office.py --method v0 --src W --tgt D --bb vgg --nlayers 16 --pretrained --model-path pretrained/vgg16.params --hybridize --train-src --end-epoch 50 --postfix p1-l2n --flip --cfg cfg/office31t.json --log-itv 10000 --l2n --lr 0.01

# Repeat 3
python train_office.py --method v0 --src A --tgt D --bb vgg --nlayers 16 --pretrained --model-path pretrained/vgg16.params --hybridize --train-src --end-epoch 20 --postfix p2 --flip --cfg cfg/office31t.json --log-itv 10000
python train_office.py --method v0 --src A --tgt W --bb vgg --nlayers 16 --pretrained --model-path pretrained/vgg16.params --hybridize --train-src --end-epoch 20 --postfix p2 --flip --cfg cfg/office31t.json --log-itv 10000
python train_office.py --method v0 --src D --tgt A --bb vgg --nlayers 16 --pretrained --model-path pretrained/vgg16.params --hybridize --train-src --end-epoch 20 --postfix p2 --flip --cfg cfg/office31t.json --log-itv 10000
python train_office.py --method v0 --src D --tgt W --bb vgg --nlayers 16 --pretrained --model-path pretrained/vgg16.params --hybridize --train-src --end-epoch 10 --postfix p2 --flip --cfg cfg/office31t.json --log-itv 10000
python train_office.py --method v0 --src W --tgt A --bb vgg --nlayers 16 --pretrained --model-path pretrained/vgg16.params --hybridize --train-src --end-epoch 20 --postfix p2 --flip --cfg cfg/office31t.json --log-itv 10000
python train_office.py --method v0 --src W --tgt D --bb vgg --nlayers 16 --pretrained --model-path pretrained/vgg16.params --hybridize --train-src --end-epoch 10 --postfix p2 --flip --cfg cfg/office31t.json --log-itv 10000

python train_office.py --method v0 --src A --tgt D --bb resnet --nlayers 101 --pretrained --model-path pretrained/resnet101_v2.params --hybridize --train-src --end-epoch 50 --postfix p2 --flip --cfg cfg/office31t.json --log-itv 10000
python train_office.py --method v0 --src A --tgt W --bb resnet --nlayers 101 --pretrained --model-path pretrained/resnet101_v2.params --hybridize --train-src --end-epoch 50 --postfix p2 --flip --cfg cfg/office31t.json --log-itv 10000
python train_office.py --method v0 --src D --tgt A --bb resnet --nlayers 101 --pretrained --model-path pretrained/resnet101_v2.params --hybridize --train-src --end-epoch 50 --postfix p2 --flip --cfg cfg/office31t.json --log-itv 10000
python train_office.py --method v0 --src D --tgt W --bb resnet --nlayers 101 --pretrained --model-path pretrained/resnet101_v2.params --hybridize --train-src --end-epoch 50 --postfix p2 --flip --cfg cfg/office31t.json --log-itv 10000
python train_office.py --method v0 --src W --tgt A --bb resnet --nlayers 101 --pretrained --model-path pretrained/resnet101_v2.params --hybridize --train-src --end-epoch 50 --postfix p2 --flip --cfg cfg/office31t.json --log-itv 10000
python train_office.py --method v0 --src W --tgt D --bb resnet --nlayers 101 --pretrained --model-path pretrained/resnet101_v2.params --hybridize --train-src --end-epoch 50 --postfix p2 --flip --cfg cfg/office31t.json --log-itv 10000

python train_office.py --method v0 --src A --tgt D --bb vgg --nlayers 16 --pretrained --model-path pretrained/vgg16.params --hybridize --train-src --end-epoch 50 --postfix p2-l2n --flip --cfg cfg/office31t.json --log-itv 10000 --l2n --lr 0.01
python train_office.py --method v0 --src A --tgt W --bb vgg --nlayers 16 --pretrained --model-path pretrained/vgg16.params --hybridize --train-src --end-epoch 50 --postfix p2-l2n --flip --cfg cfg/office31t.json --log-itv 10000 --l2n --lr 0.01
python train_office.py --method v0 --src D --tgt A --bb vgg --nlayers 16 --pretrained --model-path pretrained/vgg16.params --hybridize --train-src --end-epoch 50 --postfix p2-l2n --flip --cfg cfg/office31t.json --log-itv 10000 --l2n --lr 0.01
python train_office.py --method v0 --src D --tgt W --bb vgg --nlayers 16 --pretrained --model-path pretrained/vgg16.params --hybridize --train-src --end-epoch 50 --postfix p2-l2n --flip --cfg cfg/office31t.json --log-itv 10000 --l2n --lr 0.01
python train_office.py --method v0 --src W --tgt A --bb vgg --nlayers 16 --pretrained --model-path pretrained/vgg16.params --hybridize --train-src --end-epoch 50 --postfix p2-l2n --flip --cfg cfg/office31t.json --log-itv 10000 --l2n --lr 0.01
python train_office.py --method v0 --src W --tgt D --bb vgg --nlayers 16 --pretrained --model-path pretrained/vgg16.params --hybridize --train-src --end-epoch 50 --postfix p2-l2n --flip --cfg cfg/office31t.json --log-itv 10000 --l2n --lr 0.01
