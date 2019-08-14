#!/usr/bin/env bash
# MNIST -> SVHN
# No data augmentation
# train using only with src data
# python main_sda.py --method v0 --src MT --tgt SN --bb lenetplus --bs 256 --resize 32 --size 32 --dropout --nc 10 --lr 0.01 --end-epoch 100 --log-itv 0 --hybridize --cfg cfg/digits-a.json --train-src --postfix s-no-aug-3
# python main_sda.py --method v0 --src MT --tgt SN --bb lenetplus --bs 256 --resize 32 --size 32 --dropout --nc 10 --lr 0.01 --end-epoch 100 --log-itv 0 --hybridize --cfg cfg/digits-a.json --train-src --postfix s-no-aug-4
# python main_sda.py --method v0 --src MT --tgt SN --bb lenetplus --bs 256 --resize 32 --size 32 --dropout --nc 10 --lr 0.01 --end-epoch 100 --log-itv 0 --hybridize --cfg cfg/digits-a.json --train-src --postfix s-no-aug-5
# train only with 100 target data
# python main_sda.py --method v1 --src MT --tgt SN --bb lenetplus --bs 256 --resize 32 --size 32 --dropout --nc 10 --lr 0.01 --end-epoch 300 --log-itv 0 --hybridize --cfg cfg/digits-a.json --postfix t-no-aug-0
# python main_sda.py --method v1 --src MT --tgt SN --bb lenetplus --bs 256 --resize 32 --size 32 --dropout --nc 10 --lr 0.01 --end-epoch 300 --log-itv 0 --hybridize --cfg cfg/digits-a.json --postfix t-no-aug-1
# python main_sda.py --method v1 --src MT --tgt SN --bb lenetplus --bs 256 --resize 32 --size 32 --dropout --nc 10 --lr 0.01 --end-epoch 300 --log-itv 0 --hybridize --cfg cfg/digits-a.json --postfix t-no-aug-2
# train with src and 100 target data
# python main_sda.py --method v1 --src MT --tgt SN --bb lenetplus --bs 256 --resize 32 --size 32 --dropout --nc 10 --lr 0.01 --end-epoch 100 --log-itv 0 --hybridize --cfg cfg/digits-a.json --train-src --postfix st-no-aug-0
# python main_sda.py --method v1 --src MT --tgt SN --bb lenetplus --bs 256 --resize 32 --size 32 --dropout --nc 10 --lr 0.01 --end-epoch 100 --log-itv 0 --hybridize --cfg cfg/digits-a.json --train-src --postfix st-no-aug-1
# python main_sda.py --method v1 --src MT --tgt SN --bb lenetplus --bs 256 --resize 32 --size 32 --dropout --nc 10 --lr 0.01 --end-epoch 100 --log-itv 0 --hybridize --cfg cfg/digits-a.json --train-src --postfix st-no-aug-2
# fine-tune with 100 target data
# python main_sda.py --method v1 --src MT --tgt SN --bb lenetplus --bs 256 --resize 32 --size 32 --model-path SRC-TRAIN/lenetplus-no-aug.params --dropout --nc 10 --lr 0.01 --end-epoch 10 --log-itv 0 --hybridize --cfg cfg/digits-a.json --postfix t-tune-no-aug-0
# python main_sda.py --method v1 --src MT --tgt SN --bb lenetplus --bs 256 --resize 32 --size 32 --model-path SRC-TRAIN/lenetplus-no-aug.params --dropout --nc 10 --lr 0.01 --end-epoch 10 --log-itv 0 --hybridize --cfg cfg/digits-a.json --postfix t-tune-no-aug-1
# python main_sda.py --method v1 --src MT --tgt SN --bb lenetplus --bs 256 --resize 32 --size 32 --model-path SRC-TRAIN/lenetplus-no-aug.params --dropout --nc 10 --lr 0.01 --end-epoch 10 --log-itv 0 --hybridize --cfg cfg/digits-a.json --postfix t-tune-no-aug-2
# DSNET
python main_sda.py --method dsnet --src MT --tgt SN --bb lenetplus --bs 256 --resize 32 --size 32 --dropout --nc 10 --lr 0.01 --end-epoch 5 --log-itv 500 --hybridize --cfg cfg/digits-a.json --train-src --alpha 0.10 --postfix st-no-aug-a0.10-r3-0 --ratio 3
python main_sda.py --method dsnet --src MT --tgt SN --bb lenetplus --bs 256 --resize 32 --size 32 --dropout --nc 10 --lr 0.01 --end-epoch 5 --log-itv 500 --hybridize --cfg cfg/digits-a.json --train-src --alpha 0.10 --postfix st-no-aug-a0.10-r3-1 --ratio 3
python main_sda.py --method dsnet --src MT --tgt SN --bb lenetplus --bs 256 --resize 32 --size 32 --dropout --nc 10 --lr 0.01 --end-epoch 5 --log-itv 500 --hybridize --cfg cfg/digits-a.json --train-src --alpha 0.10 --postfix st-no-aug-a0.10-r3-2 --ratio 3

python main_sda.py --method dsnet --src MT --tgt SN --bb lenetplus --bs 256 --resize 36 --size 32 --dropout --nc 10 --lr 0.01 --end-epoch 5 --log-itv 100 --hybridize --cfg cfg/digits-a.json --train-src --alpha 0.10 --inn --random-crop --postfix st-inn-crop-a0.10-r3-0 --ratio 3
python main_sda.py --method dsnet --src MT --tgt SN --bb lenetplus --bs 256 --resize 36 --size 32 --dropout --nc 10 --lr 0.01 --end-epoch 5 --log-itv 100 --hybridize --cfg cfg/digits-a.json --train-src --alpha 0.10 --inn --random-crop --postfix st-inn-crop-a0.10-r3-1 --ratio 3
python main_sda.py --method dsnet --src MT --tgt SN --bb lenetplus --bs 256 --resize 36 --size 32 --dropout --nc 10 --lr 0.01 --end-epoch 5 --log-itv 100 --hybridize --cfg cfg/digits-a.json --train-src --alpha 0.10 --inn --random-crop --postfix st-inn-crop-a0.10-r3-2 --ratio 3

# SVHN -> MNIST
# No data augmentation
# train using only with src data
# python main_sda.py --method v0 --src SN --tgt MT --bb lenetplus --bs 256 --resize 32 --size 32 --dropout --nc 10 --lr 0.01 --end-epoch 100 --log-itv 0 --hybridize --cfg cfg/digits-a.json --train-src --postfix s-no-aug-0
# python main_sda.py --method v0 --src SN --tgt MT --bb lenetplus --bs 256 --resize 32 --size 32 --dropout --nc 10 --lr 0.01 --end-epoch 100 --log-itv 0 --hybridize --cfg cfg/digits-a.json --train-src --postfix s-no-aug-1
# python main_sda.py --method v0 --src SN --tgt MT --bb lenetplus --bs 256 --resize 32 --size 32 --dropout --nc 10 --lr 0.01 --end-epoch 100 --log-itv 0 --hybridize --cfg cfg/digits-a.json --train-src --postfix s-no-aug-2
# train only with 100 target data
# python main_sda.py --method v1 --src SN --tgt MT --bb lenetplus --bs 256 --resize 32 --size 32 --dropout --nc 10 --lr 0.01 --end-epoch 300 --log-itv 0 --hybridize --cfg cfg/digits-a.json --postfix t-no-aug-0
# python main_sda.py --method v1 --src SN --tgt MT --bb lenetplus --bs 256 --resize 32 --size 32 --dropout --nc 10 --lr 0.01 --end-epoch 300 --log-itv 0 --hybridize --cfg cfg/digits-a.json --postfix t-no-aug-1
# python main_sda.py --method v1 --src SN --tgt MT --bb lenetplus --bs 256 --resize 32 --size 32 --dropout --nc 10 --lr 0.01 --end-epoch 300 --log-itv 0 --hybridize --cfg cfg/digits-a.json --postfix t-no-aug-2
# train with src and 100 target data
# python main_sda.py --method v1 --src SN --tgt MT --bb lenetplus --bs 256 --resize 32 --size 32 --dropout --nc 10 --lr 0.01 --end-epoch 100 --log-itv 0 --hybridize --cfg cfg/digits-a.json --train-src --postfix st-no-aug-0
# python main_sda.py --method v1 --src SN --tgt MT --bb lenetplus --bs 256 --resize 32 --size 32 --dropout --nc 10 --lr 0.01 --end-epoch 100 --log-itv 0 --hybridize --cfg cfg/digits-a.json --train-src --postfix st-no-aug-1
# python main_sda.py --method v1 --src SN --tgt MT --bb lenetplus --bs 256 --resize 32 --size 32 --dropout --nc 10 --lr 0.01 --end-epoch 100 --log-itv 0 --hybridize --cfg cfg/digits-a.json --train-src --postfix st-no-aug-2
# fine-tune with 100 target data
# python main_sda.py --method v1 --src SN --tgt MT --bb lenetplus --bs 256 --resize 32 --size 32 --model-path SRC-TRAIN/lenetplus-no-aug.params --dropout --nc 10 --lr 0.01 --end-epoch 30 --log-itv 0 --hybridize --cfg cfg/digits-a.json --postfix t-tune-no-aug-0
# python main_sda.py --method v1 --src SN --tgt MT --bb lenetplus --bs 256 --resize 32 --size 32 --model-path SRC-TRAIN/lenetplus-no-aug.params --dropout --nc 10 --lr 0.01 --end-epoch 30 --log-itv 0 --hybridize --cfg cfg/digits-a.json --postfix t-tune-no-aug-1
# python main_sda.py --method v1 --src SN --tgt MT --bb lenetplus --bs 256 --resize 32 --size 32 --model-path SRC-TRAIN/lenetplus-no-aug.params --dropout --nc 10 --lr 0.01 --end-epoch 30 --log-itv 0 --hybridize --cfg cfg/digits-a.json --postfix t-tune-no-aug-2
# DSNET
#python main_sda.py --method dsnet --src SN --tgt MT --bb lenetplus --bs 256 --resize 32 --size 32 --dropout --nc 10 --lr 0.01 --end-epoch 5 --log-itv 500 --hybridize --cfg cfg/digits-a.json --train-src --alpha 0.10 --postfix st-no-aug-a0.10-0 --ratio 3
#python main_sda.py --method dsnet --src SN --tgt MT --bb lenetplus --bs 256 --resize 32 --size 32 --dropout --nc 10 --lr 0.01 --end-epoch 5 --log-itv 500 --hybridize --cfg cfg/digits-a.json --train-src --alpha 0.10 --postfix st-no-aug-a0.10-1 --ratio 3
#python main_sda.py --method dsnet --src SN --tgt MT --bb lenetplus --bs 256 --resize 32 --size 32 --dropout --nc 10 --lr 0.01 --end-epoch 5 --log-itv 500 --hybridize --cfg cfg/digits-a.json --train-src --alpha 0.10 --postfix st-no-aug-a0.10-2 --ratio 3
