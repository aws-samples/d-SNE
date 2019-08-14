#!/usr/bin/env bash
# No data augmentation
# train using only with src data
#python main_sda.py --method v0 --src MT --tgt US --bb lenetplus --bs 256 --resize 32 --size 32 --dropout --nc 10 --lr 0.01 --end-epoch 30 --log-itv 0 --hybridize --cfg cfg/digits-a.json --train-src --postfix s-no-aug-0
#python main_sda.py --method v0 --src MT --tgt US --bb lenetplus --bs 256 --resize 32 --size 32 --dropout --nc 10 --lr 0.01 --end-epoch 30 --log-itv 0 --hybridize --cfg cfg/digits-a.json --train-src --postfix s-no-aug-1
#python main_sda.py --method v0 --src MT --tgt US --bb lenetplus --bs 256 --resize 32 --size 32 --dropout --nc 10 --lr 0.01 --end-epoch 30 --log-itv 0 --hybridize --cfg cfg/digits-a.json --train-src --postfix s-no-aug-2
# train only with 100 target data
#python main_sda.py --method v1 --src MT --tgt US --bb lenetplus --bs 256 --resize 32 --size 32 --dropout --nc 10 --lr 0.01 --end-epoch 300 --log-itv 0 --hybridize --cfg cfg/digits-a.json --postfix t-no-aug-0
#python main_sda.py --method v1 --src MT --tgt US --bb lenetplus --bs 256 --resize 32 --size 32 --dropout --nc 10 --lr 0.01 --end-epoch 300 --log-itv 0 --hybridize --cfg cfg/digits-a.json --postfix t-no-aug-1
#python main_sda.py --method v1 --src MT --tgt US --bb lenetplus --bs 256 --resize 32 --size 32 --dropout --nc 10 --lr 0.01 --end-epoch 300 --log-itv 0 --hybridize --cfg cfg/digits-a.json --postfix t-no-aug-2
# train with src and 100 target data
#python main_sda.py --method v1 --src MT --tgt US --bb lenetplus --bs 256 --resize 32 --size 32 --dropout --nc 10 --lr 0.01 --end-epoch 30 --log-itv 0 --hybridize --cfg cfg/digits-a.json --train-src --postfix st-no-aug-0
#python main_sda.py --method v1 --src MT --tgt US --bb lenetplus --bs 256 --resize 32 --size 32 --dropout --nc 10 --lr 0.01 --end-epoch 30 --log-itv 0 --hybridize --cfg cfg/digits-a.json --train-src --postfix st-no-aug-1
#python main_sda.py --method v1 --src MT --tgt US --bb lenetplus --bs 256 --resize 32 --size 32 --dropout --nc 10 --lr 0.01 --end-epoch 30 --log-itv 0 --hybridize --cfg cfg/digits-a.json --train-src --postfix st-no-aug-2
# fine-tune with 100 target data
#python main_sda.py --method v1 --src MT --tgt US --bb lenetplus --bs 256 --resize 32 --size 32 --model-path SRC-TRAIN/lenetplus-no-aug.params --dropout --nc 10 --lr 0.01 --end-epoch 10 --log-itv 0 --hybridize --cfg cfg/digits-a.json --postfix t-tune-no-aug-0
#python main_sda.py --method v1 --src MT --tgt US --bb lenetplus --bs 256 --resize 32 --size 32 --model-path SRC-TRAIN/lenetplus-no-aug.params --dropout --nc 10 --lr 0.01 --end-epoch 10 --log-itv 0 --hybridize --cfg cfg/digits-a.json --postfix t-tune-no-aug-1
#python main_sda.py --method v1 --src MT --tgt US --bb lenetplus --bs 256 --resize 32 --size 32 --model-path SRC-TRAIN/lenetplus-no-aug.params --dropout --nc 10 --lr 0.01 --end-epoch 10 --log-itv 0 --hybridize --cfg cfg/digits-a.json --postfix t-tune-no-aug-2
# DSNET
python main_sda.py --method dsnet --src MT --tgt US --bb lenetplus --bs 256 --resize 32 --size 32 --dropout --nc 10 --lr 0.01 --end-epoch 5 --log-itv 100 --hybridize --cfg cfg/digits-a.json --train-src --alpha 0.25 --ratio=3 --postfix st-no-aug-a0.25-0
python main_sda.py --method dsnet --src MT --tgt US --bb lenetplus --bs 256 --resize 32 --size 32 --dropout --nc 10 --lr 0.01 --end-epoch 5 --log-itv 100 --hybridize --cfg cfg/digits-a.json --train-src --alpha 0.25 --ratio=3 --postfix st-no-aug-a0.25-1
python main_sda.py --method dsnet --src MT --tgt US --bb lenetplus --bs 256 --resize 32 --size 32 --dropout --nc 10 --lr 0.01 --end-epoch 5 --log-itv 100 --hybridize --cfg cfg/digits-a.json --train-src --alpha 0.25 --ratio=3 --postfix st-no-aug-a0.25-2

# US -> MT
#python main_sda.py --method v0 --src US --tgt MT --bb lenetplus --bs 256 --resize 32 --size 32 --dropout --nc 10 --lr 0.01 --end-epoch 30 --log-itv 0 --hybridize --cfg cfg/digits-a.json --train-src --postfix s-no-aug-0
#python main_sda.py --method v0 --src US --tgt MT --bb lenetplus --bs 256 --resize 32 --size 32 --dropout --nc 10 --lr 0.01 --end-epoch 30 --log-itv 0 --hybridize --cfg cfg/digits-a.json --train-src --postfix s-no-aug-1
#python main_sda.py --method v0 --src US --tgt MT --bb lenetplus --bs 256 --resize 32 --size 32 --dropout --nc 10 --lr 0.01 --end-epoch 30 --log-itv 0 --hybridize --cfg cfg/digits-a.json --train-src --postfix s-no-aug-2
# train only with 100 target data
#python main_sda.py --method v1 --src US --tgt MT --bb lenetplus --bs 256 --resize 32 --size 32 --dropout --nc 10 --lr 0.01 --end-epoch 300 --log-itv 0 --hybridize --cfg cfg/digits-a.json --postfix t-no-aug-0
#python main_sda.py --method v1 --src US --tgt MT --bb lenetplus --bs 256 --resize 32 --size 32 --dropout --nc 10 --lr 0.01 --end-epoch 300 --log-itv 0 --hybridize --cfg cfg/digits-a.json --postfix t-no-aug-1
#python main_sda.py --method v1 --src US --tgt MT --bb lenetplus --bs 256 --resize 32 --size 32 --dropout --nc 10 --lr 0.01 --end-epoch 300 --log-itv 0 --hybridize --cfg cfg/digits-a.json --postfix t-no-aug-2
# train with src and 100 target data
#python main_sda.py --method v1 --src US --tgt MT --bb lenetplus --bs 256 --resize 32 --size 32 --dropout --nc 10 --lr 0.01 --end-epoch 30 --log-itv 0 --hybridize --cfg cfg/digits-a.json --train-src --postfix st-no-aug-0
#python main_sda.py --method v1 --src US --tgt MT --bb lenetplus --bs 256 --resize 32 --size 32 --dropout --nc 10 --lr 0.01 --end-epoch 30 --log-itv 0 --hybridize --cfg cfg/digits-a.json --train-src --postfix st-no-aug-1
#python main_sda.py --method v1 --src US --tgt MT --bb lenetplus --bs 256 --resize 32 --size 32 --dropout --nc 10 --lr 0.01 --end-epoch 30 --log-itv 0 --hybridize --cfg cfg/digits-a.json --train-src --postfix st-no-aug-2
# fine-tune with 100 target data
#python main_sda.py --method v1 --src US --tgt MT --bb lenetplus --bs 256 --resize 32 --size 32 --model-path SRC-TRAIN/lenetplus-no-aug.params --dropout --nc 10 --lr 0.01 --end-epoch 10 --log-itv 0 --hybridize --cfg cfg/digits-a.json --postfix t-tune-no-aug-0
#python main_sda.py --method v1 --src US --tgt MT --bb lenetplus --bs 256 --resize 32 --size 32 --model-path SRC-TRAIN/lenetplus-no-aug.params --dropout --nc 10 --lr 0.01 --end-epoch 10 --log-itv 0 --hybridize --cfg cfg/digits-a.json --postfix t-tune-no-aug-1
#python main_sda.py --method v1 --src US --tgt MT --bb lenetplus --bs 256 --resize 32 --size 32 --model-path SRC-TRAIN/lenetplus-no-aug.params --dropout --nc 10 --lr 0.01 --end-epoch 10 --log-itv 0 --hybridize --cfg cfg/digits-a.json --postfix t-tune-no-aug-2
# DSNET
python main_sda.py --method dsnet --src US --tgt MT --bb lenetplus --bs 256 --resize 32 --size 32 --dropout --nc 10 --lr 0.01 --end-epoch 5 --log-itv 100 --hybridize --cfg cfg/digits-a.json --train-src --alpha 0.25 --ratio=3 --postfix st-no-aug-a0.25-0
python main_sda.py --method dsnet --src US --tgt MT --bb lenetplus --bs 256 --resize 32 --size 32 --dropout --nc 10 --lr 0.01 --end-epoch 5 --log-itv 100 --hybridize --cfg cfg/digits-a.json --train-src --alpha 0.25 --ratio=3 --postfix st-no-aug-a0.25-1
python main_sda.py --method dsnet --src US --tgt MT --bb lenetplus --bs 256 --resize 32 --size 32 --dropout --nc 10 --lr 0.01 --end-epoch 5 --log-itv 100 --hybridize --cfg cfg/digits-a.json --train-src --alpha 0.25 --ratio=3 --postfix st-no-aug-a0.25-2