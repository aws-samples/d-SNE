"""
Logging module for domain adaptation
"""
import os
import logging
import shutil


class Logger(object):
    """
    Logger to console and file
    """
    def __init__(self, log_dir, name, rm_exist=True):

        logger_file = os.path.join(log_dir, name + '.log')

        if rm_exist:
            # rm_existing(log_dir)
            rm_mk_dir(log_dir)

        mk_pardir(logger_file)

        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)
        # create file handler
        fh = logging.FileHandler(logger_file)
        fh.setLevel(logging.DEBUG)
        # create console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)
        # create formatter and add it to the handler
        formatter = logging.Formatter('%(asctime)s:%(message)s')
        ch.setFormatter(formatter)
        fh.setFormatter(formatter)
        # add the handler to logger
        self.logger.addHandler(ch)
        self.logger.addHandler(fh)

    def update_scalar(self, name, value, level='DEBUG'):
        if level == 'DEBUG':
            self.logger.debug('%s=%f' % (name, value))
        elif level == 'WARNING':
            self.logger.warning('%s=%f' % (name, value))
        elif level == 'ERROR':
            self.logger.error('%s=%f' % (name, value))
        else:
            self.logger.info('%s=%f' % (name, value))

    def update_dict(self, d, level='DEBUG'):
        s = '\n'.join(['%s=%s' % (k, str(v)) for k, v in d.items()])
        if level == 'DEBUG':
            self.logger.debug(s)
        elif level == 'WARNING':
            self.logger.warning(s)
        elif level == 'ERROR':
            self.logger.error(s)
        else:
            self.logger.info(s)


def rm_existing(path):
    """
    remove existing file
    :param path: file path
    :return:
    """
    if os.path.exists(path):
        os.remove(path)


def rm_mk_dir(dir_path):
    if os.path.exists(dir_path):
        shutil.rmtree(dir_path)

    os.makedirs(dir_path)


def mk_pardir(path):
    """
    make parent directory
    :param path: file path
    :return:
    """
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
