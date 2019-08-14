#!/usr/bin/env bash
# no augmentation
# src + target
python main_ssda.py --method mts --src MT --tgt US --bb lenetplus --bs 256 --resize 32 --size 32 --postfix s-no-aug-b50-0 --lr 0.01 --end-epoch 30 --log-itv 0 --cfg cfg/digits-a.json --hybridize --nc 10 --dropout --train-src --beta 50
python main_ssda.py --method mts --src MT --tgt US --bb lenetplus --bs 256 --resize 32 --size 32 --postfix s-no-aug-b50-1 --lr 0.01 --end-epoch 30 --log-itv 0 --cfg cfg/digits-a.json --hybridize --nc 10 --dropout --train-src --beta 50
python main_ssda.py --method mts --src MT --tgt US --bb lenetplus --bs 256 --resize 32 --size 32 --postfix s-no-aug-b50-2 --lr 0.01 --end-epoch 30 --log-itv 0 --cfg cfg/digits-a.json --hybridize --nc 10 --dropout --train-src --beta 50
# 100 target + target
python main_ssda.py --method mtt --src MT --tgt US --bb lenetplus --bs 256 --resize 32 --size 32 --postfix t-no-aug-b50-0 --lr 0.01 --end-epoch 30 --log-itv 0 --cfg cfg/digits-a.json --hybridize --nc 10 --dropout --beta 50
python main_ssda.py --method mtt --src MT --tgt US --bb lenetplus --bs 256 --resize 32 --size 32 --postfix t-no-aug-b50-1 --lr 0.01 --end-epoch 30 --log-itv 0 --cfg cfg/digits-a.json --hybridize --nc 10 --dropout --beta 50
python main_ssda.py --method mtt --src MT --tgt US --bb lenetplus --bs 256 --resize 32 --size 32 --postfix t-no-aug-b50-2 --lr 0.01 --end-epoch 30 --log-itv 0 --cfg cfg/digits-a.json --hybridize --nc 10 --dropout --beta 50
# src + 100 target + target
python main_ssda.py --method mtt --src MT --tgt US --bb lenetplus --bs 256 --resize 32 --size 32 --postfix st-no-aug-b50-0 --lr 0.01 --end-epoch 30 --log-itv 0 --cfg cfg/digits-a.json --hybridize --nc 10 --dropout --train-src --beta 50
python main_ssda.py --method mtt --src MT --tgt US --bb lenetplus --bs 256 --resize 32 --size 32 --postfix st-no-aug-b50-1 --lr 0.01 --end-epoch 30 --log-itv 0 --cfg cfg/digits-a.json --hybridize --nc 10 --dropout --train-src --beta 50
python main_ssda.py --method mtt --src MT --tgt US --bb lenetplus --bs 256 --resize 32 --size 32 --postfix st-no-aug-b50-2 --lr 0.01 --end-epoch 30 --log-itv 0 --cfg cfg/digits-a.json --hybridize --nc 10 --dropout --train-src --beta 50
# dsnet
python main_ssda.py --method mtd --cfg cfg/digits-a.json --src MT --tgt US --bb lenetplus --bs 256 --resize 32 --size 32 --model-path DSNET/lenetplus-no-aug-acc=99.09.params --rampup-epoch 1 --lr 0.01 --end-epoch 80 --log-itv 0 --nc 10 --dropout --beta 50 --train-src --postfix st-no-aug-0
python main_ssda.py --method mtd --cfg cfg/digits-a.json --src MT --tgt US --bb lenetplus --bs 256 --resize 32 --size 32 --model-path DSNET/lenetplus-no-aug-acc=99.09.params --rampup-epoch 1 --lr 0.01 --end-epoch 80 --log-itv 0 --nc 10 --dropout --beta 50 --train-src --postfix st-no-aug-1
python main_ssda.py --method mtd --cfg cfg/digits-a.json --src MT --tgt US --bb lenetplus --bs 256 --resize 32 --size 32 --model-path DSNET/lenetplus-no-aug-acc=99.09.params --rampup-epoch 1 --lr 0.01 --end-epoch 80 --log-itv 0 --nc 10 --dropout --beta 50 --train-src --postfix st-no-aug-2
