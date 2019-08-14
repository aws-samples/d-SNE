import subprocess
import argparse
import os


def main():
    if args.year == '2017':
        url = 'http://csr.bu.edu/ftp/visda17/clf'
        d = args.dir + '17'
    else:
        url = 'http://csr.bu.edu/ftp/visda/2018/openset'
        d = args.dir + '18'

    if not os.path.exists(d):
        os.makedirs(d)

    # download train
    subprocess.check_output(
        '''
        cd %s
        wget -c %s
        ''' % (d, os.path.join(url, 'train.tar')),
        shell=True)

    # download validation
    subprocess.check_output(
        '''
        cd %s
        wget -c %s
        ''' % (d, os.path.join(url, 'validation.tar')),
        shell=True)

    if args.year == '2017':
        # download test
        subprocess.check_output(
            '''
            cd %s
            wget -c %s
            ''' % (d, os.path.join(url, 'test.tar')),
            shell=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('year', help='Year of VisDA challenge')
    parser.add_argument('--dir', default='VisDA', help='directory')

    args = parser.parse_args()

    main()
