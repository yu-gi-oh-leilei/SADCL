import math
import os, sys
import random
import time
import json
# os.environ['CUDA_VISIBLE_DEVICES'] ='2,3'

import _init_paths
from config import parser_args


from utils.misc import init_distributed_and_seed
from utils.util import show_args, init_logeger
from main_worker import main_worker

def get_args():
    args = parser_args()
    return args

def main():
    args = get_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpus)

    # init distributed and seed
    init_distributed_and_seed(args)
    
    # init logeger and show config
    logger = init_logeger(args)
    show_args(args, logger)

    return main_worker(args, logger)

if __name__ == '__main__':
    main()