"""
This script is used for training on semi-supervised domain adaptation on the image classification
"""

from utils.parse_args import parse_args_ssda
from train_val.training_ssda import MeanTeacher, MeanTeacherDSNET, MeanTeacherDSNETv2


def main():
    if args.method.upper() == 'MTS':
        model = MeanTeacher(args, use_teacher=True, train_tgt=False)
    elif args.method.upper() == 'MTT':
        model = MeanTeacher(args, use_teacher=True, train_tgt=True)
    elif args.method.upper() == 'MTD':
        model = MeanTeacherDSNET(args)
    elif args.method.upper() == 'MTD2':
        model = MeanTeacherDSNETv2(args)
    else:
        raise NotImplementedError

    if args.training:
        model.train()
    else:
        model.test()


if __name__ == '__main__':
    args = parse_args_ssda()

    main()
