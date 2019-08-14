#!/usr/bin/env bash
# Experiment on MNIST -> USPS
# Setting 2000 samples in training
# train src
# python main_sda.py --cfg cfg/digits-s-1.json --src MT --tgt US --method v0 --nc 10 --size 32 --bb lenetplus --dropout --train-src --lr 0.01 --gpus 0 --end-epoch 25 --log-itv 0 --postfix s0
# train with dSNEt
#python main_sda.py --cfg cfg/digits-s-1.json --src MT --tgt US --method dsnet --nc 10 --size 32 --bb conv2 --dropout --train-src --lr 0.01 --gpus 0 --end-epoch 50 --log-itv 0 --postfix s1-0.75 --embed-size 80 --fn --alpha 0.75
#python main_sda.py --cfg cfg/digits-s-3.json --src MT --tgt US --method dsnet --nc 10 --size 32 --bb conv2 --dropout --train-src --lr 0.01 --gpus 0 --end-epoch 50 --log-itv 0 --postfix s3-0.75 --embed-size 80 --fn --alpha 0.75
#python main_sda.py --cfg cfg/digits-s-5.json --src MT --tgt US --method dsnet --nc 10 --size 32 --bb conv2 --dropout --train-src --lr 0.01 --gpus 0 --end-epoch 50 --log-itv 0 --postfix s5-0.75 --embed-size 80 --fn --alpha 0.75
#python main_sda.py --cfg cfg/digits-s-7.json --src MT --tgt US --method dsnet --nc 10 --size 32 --bb conv2 --dropout --train-src --lr 0.01 --gpus 0 --end-epoch 50 --log-itv 0 --postfix s7-0.75 --embed-size 80 --fn --alpha 0.75
#
# python main_sda.py --cfg cfg/digits-s-1.json --src MT --tgt US --method dsnet --nc 10 --size 32 --bb conv2 --dropout --train-src --lr 0.01 --gpus 0 --end-epoch 50 --log-itv 0 --postfix s1-0.5-0 --embed-size 80 --fn --alpha 0.5
# python main_sda.py --cfg cfg/digits-s-3.json --src MT --tgt US --method dsnet --nc 10 --size 32 --bb conv2 --dropout --train-src --lr 0.01 --gpus 0 --end-epoch 50 --log-itv 0 --postfix s3-0.5-0 --embed-size 80 --fn --alpha 0.5
# python main_sda.py --cfg cfg/digits-s-5.json --src MT --tgt US --method dsnet --nc 10 --size 32 --bb conv2 --dropout --train-src --lr 0.01 --gpus 0 --end-epoch 50 --log-itv 0 --postfix s5-0.5-0 --embed-size 80 --fn --alpha 0.5
# python main_sda.py --cfg cfg/digits-s-7.json --src MT --tgt US --method dsnet --nc 10 --size 32 --bb conv2 --dropout --train-src --lr 0.01 --gpus 0 --end-epoch 50 --log-itv 0 --postfix s7-0.5-0 --embed-size 80 --fn --alpha 0.5

#
#python main_sda.py --cfg cfg/digits-s-1.json --src MT --tgt US --method dsnet --nc 10 --size 32 --bb conv2 --dropout --train-src --lr 0.01 --gpus 0 --end-epoch 50 --log-itv 0 --postfix s1-0.25 --embed-size 80 --fn --alpha 0.25
#python main_sda.py --cfg cfg/digits-s-3.json --src MT --tgt US --method dsnet --nc 10 --size 32 --bb conv2 --dropout --train-src --lr 0.01 --gpus 0 --end-epoch 50 --log-itv 0 --postfix s3-0.25 --embed-size 80 --fn --alpha 0.25
#python main_sda.py --cfg cfg/digits-s-5.json --src MT --tgt US --method dsnet --nc 10 --size 32 --bb conv2 --dropout --train-src --lr 0.01 --gpus 0 --end-epoch 50 --log-itv 0 --postfix s5-0.25 --embed-size 80 --fn --alpha 0.25
#python main_sda.py --cfg cfg/digits-s-7.json --src MT --tgt US --method dsnet --nc 10 --size 32 --bb conv2 --dropout --train-src --lr 0.01 --gpus 0 --end-epoch 50 --log-itv 0 --postfix s7-0.25 --embed-size 80 --fn --alpha 0.25

#python main_sda.py --cfg cfg/digits-s-1.json --src MT --tgt US --method dsnet --nc 10 --size 32 --bb conv2 --dropout --train-src --lr 0.01 --gpus 0 --end-epoch 50 --log-itv 0 --postfix s1-0.5-1 --embed-size 80 --fn --alpha 0.5
#python main_sda.py --cfg cfg/digits-s-3.json --src MT --tgt US --method dsnet --nc 10 --size 32 --bb conv2 --dropout --train-src --lr 0.01 --gpus 0 --end-epoch 50 --log-itv 0 --postfix s3-0.5-1 --embed-size 80 --fn --alpha 0.5
#python main_sda.py --cfg cfg/digits-s-5.json --src MT --tgt US --method dsnet --nc 10 --size 32 --bb conv2 --dropout --train-src --lr 0.01 --gpus 0 --end-epoch 50 --log-itv 0 --postfix s5-0.5-1 --embed-size 80 --fn --alpha 0.5
#python main_sda.py --cfg cfg/digits-s-7.json --src MT --tgt US --method dsnet --nc 10 --size 32 --bb conv2 --dropout --train-src --lr 0.01 --gpus 0 --end-epoch 50 --log-itv 0 --postfix s7-0.5-1 --embed-size 80 --fn --alpha 0.5

# python main_sda.py --cfg cfg/digits-s-1.json --src MT --tgt US --method dsnet --nc 10 --size 32 --bb conv2 --dropout --train-src --lr 0.01 --gpus 0 --end-epoch 50 --log-itv 0 --postfix s1-0.5-2 --embed-size 80 --fn --alpha 0.5
# python main_sda.py --cfg cfg/digits-s-3.json --src MT --tgt US --method dsnet --nc 10 --size 32 --bb conv2 --dropout --train-src --lr 0.01 --gpus 0 --end-epoch 50 --log-itv 0 --postfix s3-0.5-2 --embed-size 80 --fn --alpha 0.5
# python main_sda.py --cfg cfg/digits-s-5.json --src MT --tgt US --method dsnet --nc 10 --size 32 --bb conv2 --dropout --train-src --lr 0.01 --gpus 0 --end-epoch 50 --log-itv 0 --postfix s5-0.5-2 --embed-size 80 --fn --alpha 0.5
# python main_sda.py --cfg cfg/digits-s-7.json --src MT --tgt US --method dsnet --nc 10 --size 32 --bb conv2 --dropout --train-src --lr 0.01 --gpus 0 --end-epoch 50 --log-itv 0 --postfix s7-0.5-2 --embed-size 80 --fn --alpha 0.5

#python main_sda.py --cfg cfg/digits-s-1.json --src MT --tgt US --method dsnet --nc 10 --size 32 --bb conv2 --dropout --train-src --lr 0.1 --gpus 0 --end-epoch 100 --lr-epoch 50 --log-itv 0 --postfix s1-0.5-l2n-0 --embed-size 80 --l2n --alpha 0.5
#python main_sda.py --cfg cfg/digits-s-3.json --src MT --tgt US --method dsnet --nc 10 --size 32 --bb conv2 --dropout --train-src --lr 0.1 --gpus 0 --end-epoch 100 --lr-epoch 50 --log-itv 0 --postfix s3-0.5-l2n-0 --embed-size 80 --l2n --alpha 0.5
#python main_sda.py --cfg cfg/digits-s-5.json --src MT --tgt US --method dsnet --nc 10 --size 32 --bb conv2 --dropout --train-src --lr 0.1 --gpus 0 --end-epoch 100 --lr-epoch 50 --log-itv 0 --postfix s5-0.5-l2n-0 --embed-size 80 --l2n --alpha 0.5
#python main_sda.py --cfg cfg/digits-s-7.json --src MT --tgt US --method dsnet --nc 10 --size 32 --bb conv2 --dropout --train-src --lr 0.1 --gpus 0 --end-epoch 100 --lr-epoch 50 --log-itv 0 --postfix s7-0.5-l2n-0 --embed-size 80 --l2n --alpha 0.5
