#!/usr/bin/env bash
cd datasets

# Download VisDA17
if [ ! -d "VisDA17" ]; then
    mkdir "VisDA17"
fi

wget https://s3-us-west-2.amazonaws.com/domain-adaptation-exps/datasets/VisDA17/src-0.idx -O VisDA17/src-0.idx
wget https://s3-us-west-2.amazonaws.com/domain-adaptation-exps/datasets/VisDA17/src-0.lst -O VisDA17/src-0.lst
wget https://s3-us-west-2.amazonaws.com/domain-adaptation-exps/datasets/VisDA17/src-0.rec -O VisDA17/src-0.rec
wget https://s3-us-west-2.amazonaws.com/domain-adaptation-exps/datasets/VisDA17/tgt-120.idx -O VisDA17/tgt-120.idx
wget https://s3-us-west-2.amazonaws.com/domain-adaptation-exps/datasets/VisDA17/tgt-120.lst -O VisDA17/tgt-120.lst
wget https://s3-us-west-2.amazonaws.com/domain-adaptation-exps/datasets/VisDA17/tgt-120.rec -O VisDA17/tgt-120.rec
wget https://s3-us-west-2.amazonaws.com/domain-adaptation-exps/datasets/VisDA17/tgt-55268.idx -O VisDA17/tgt-55268.idx
wget https://s3-us-west-2.amazonaws.com/domain-adaptation-exps/datasets/VisDA17/tgt-55268.lst -O VisDA17/tgt-55268.lst
wget https://s3-us-west-2.amazonaws.com/domain-adaptation-exps/datasets/VisDA17/tgt-55268.rec -O VisDA17/tgt-55268.rec
