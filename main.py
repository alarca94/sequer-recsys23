import torch
import yaml
import sys
import argparse
import logging

from models import Trainer
from utils.data_utils import now_time, init_seed
from utils.constants import *


def main(args):
    if args.log_to_file:
        log_f = f'{args.dataset}_{args.model_name}{args.model_suffix}_0_{ALL_SEEDS.index(args.seed)}_{DATA_MODE}.txt'
        handlers = [logging.FileHandler(filename=os.path.join(LOG_PATH, log_f), mode='w')]
    else:
        handlers = [logging.StreamHandler(stream=sys.stdout)]

    logging.basicConfig(level=logging.DEBUG,
                        format='%(levelname)s - %(message)s',
                        handlers=handlers)

    with open(os.path.join(CONFIG_PATH, f'{args.model_name}{args.model_suffix}.yaml'), 'r') as f:
        cfg = yaml.load(f, yaml.FullLoader)

    # Set the random seed manually for reproducibility.
    init_seed(args.seed, reproducibility=True)
    if torch.cuda.is_available():
        if not args.cuda:
            logging.info(now_time() + 'WARNING: You have a CUDA device, so you should probably run with --cuda')
    device = torch.device('cuda' if args.cuda else 'cpu')

    if not os.path.exists(CKPT_PATH):
        os.makedirs(CKPT_PATH)
    pred_file = PRED_F.split('.')
    pred_file[0] += f'_{args.dataset}_{args.fold}_{args.model_name}{args.model_suffix}'
    prediction_path = os.path.join(CKPT_PATH, DATA_MODE, '.'.join(pred_file))

    if args.test:
        cfg['epochs'] = 1

    args.no_generate = args.no_generate | args.test

    logging.info('-' * 40 + 'ARGUMENTS' + '-' * 40)
    for arg in vars(args):
        logging.info('{:40} {}'.format(arg, getattr(args, arg)))
    for arg in cfg:
        logging.info('{:40} {}'.format(arg, cfg[arg]))
    logging.info('-' * 40 + 'ARGUMENTS' + '-' * 40)

    trainer = Trainer(args, cfg, device, prediction_path)
    if not args.load_checkpoint:
        trainer.train()
    trainer.test()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SEQuence-Aware Explainable Recommendation (SEQUER)')
    parser.add_argument('--model-name', type=str, default='sequer',
                        help='model name (sequer)')
    parser.add_argument('--model-suffix', type=str, default='',
                        help='model suffix for different model configs (_mp, _r, +, etc.)')
    parser.add_argument('--dataset', type=str, default=None,
                        help='dataset name (amazon_movies, yelp, tripadvisor)')
    parser.add_argument('--fold', type=str, default=0,
                        help='data partition index')
    parser.add_argument('--cuda', action='store_true',
                        help='use CUDA')
    parser.add_argument('--test', action='store_true',
                        help='use a small fraction of the data')
    parser.add_argument('--no-save-results', dest="save_results", default=True, action='store_false',
                        help='whether results will be saved or not')
    parser.add_argument('--no-generate', action='store_true',
                        help='whether generated text will be saved to log file or not')
    parser.add_argument('--load-checkpoint', action='store_true',
                        help='Whether to load an existing checkpoint and evaluate it')
    parser.add_argument('--log-to-file', action='store_true',
                        help='Whether to print directly to a log file')
    parser.add_argument('--seed', type=int, default=RNG_SEED,
                        help='seed for reproducibility')

    args = parser.parse_args()
    if args.dataset is None:
        parser.error('--dataset should be provided for loading data')
    elif args.dataset not in DATA_PATHS:
        parser.error(
            f'--dataset supported values are: {", ".join(list(DATA_PATHS.keys()))} -- Provided value: {args.dataset}')

    main(args)
