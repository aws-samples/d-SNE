"""
This scripts for main function of supervised domain adaptation on image classification
"""

from utils.parse_args import parse_args_sda
from train_val.training_sda import ClsModel, CCSA, dSNE


def main():
    """
    Main function
    CCSA: ICCV 17 model
    V0: train on source and test on target
    V1: train on source and target
    dsne: dSNE model
    dsnet: dSNE-Triplet model
    :return:
    """
    if args.method == 'v0':
        model = ClsModel(args, train_tgt=False)
    elif args.method == 'v1':
        model = ClsModel(args, train_tgt=True)
    elif args.method == 'ccsa':
        model = CCSA(args)
    elif args.method == 'dsne':
        model = dSNE(args)
    else:
        raise NotImplementedError

    if args.training:
        model.train()
    else:
        model.test()


if __name__ == '__main__':
    args = parse_args_sda()

    main()
