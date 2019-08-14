#!/usr/bin/env bash
# use src data to train the model
python main_sda.py --method v0 --cfg cfg/visda17.json --postfix src-train --bs 32 --log-itv 500 --hybridize --flip --random-crop --random-color --end-epoch 20 --train-src
# use imagenet pre-trained model and tune on the source data
python main_sda.py --method v0 --cfg cfg/visda17.json --postfix src-tune --bs 32 --log-itv 500 --bb --pretrained --model-path pretrained/resnet152_v2.params --hybridize --flip --random-crop --random-color --lr 0.001 --end-epoch 20 --train-src
# use imagenet pre-trained model and tune on the target data
python main_sda.py --method v1 --cfg cfg/visda17.json --postfix tgt-tune --bs 32 --log-itv 0 --pretrained --model-path pretrained/resnet152_v2.params --hybridize --flip --random-crop --random-color --color-jitter 0.4 --eval-epoch 40 --end-epoch 45
# use tuned source model and fine-tune on the target data
python main_sda.py --method v1 --cfg cfg/visda17.json --postfix tgt-src-tune --bs 32 --log-itv 0 --model-path SRC-TUNE/resnet152_v2.params --hybridize --flip --random-crop --random-color --lr 0.001 --end-epoch 20
# use trained source model and fine-tune on the target data
python main_sda.py --method v1 --cfg cfg/visda17.json --postfix tgt-src-train --bs 32 --log-itv 0 --model-path SRC-TRAIN/resnet152_v2.params --hybridize --flip --random-crop --random-color --lr 0.001 --end-epoch 20
# train DSNE-T dataset
python main_sda.py --method dsnet --cfg cfg/visda17.json --postfix 0 --bs 48 --log-itv 100 --hybridize --flip --random-crop --random-color --lr 0.001 --end-epoch 50 --fn --pretrained --model-path pretrained/resnet152_v2.params --alpha 0.25
