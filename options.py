import argparse
from loguru import logger

parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', type=str, default='./models')
parser.add_argument('--data_dir', type=str, default='./data')
parser.add_argument('--output_dir', type=str, default='./data')
parser.add_argument('--save_result_dir', type=str, default='./data')  # for eval
# model
parser.add_argument("--model", type=str, default='bert')
parser.add_argument("--mentions_max_length", type=int, default=40)
parser.add_argument("--candidates_max_length", type=int, default=40)
parser.add_argument("--test_model", type=str, default=None)
parser.add_argument("--load_model", type=str, default=None)
parser.add_argument("--embed_size", type=int, default=768)

# training
parser.add_argument("--n_epochs", type=int, default=20)
parser.add_argument("--dropout", type=float, default=0.5)
parser.add_argument("--dir_name", type=str, default='bert2lr5epoch20hard20')
parser.add_argument("--agg_mode", type=str, default='cls')
parser.add_argument("--train_top_k", type=int, default=20)
parser.add_argument("--top_k", type=int, default=10)
parser.add_argument("--hard", type=int, default=20)
parser.add_argument("--batch_size", type=int, default=4)
parser.add_argument("--lr", type=float, default=2e-5)
parser.add_argument("--weight_decay", type=float, default=0)
parser.add_argument("--gpu", type=int, default=1, help='-1 if not use gpu, >=0 if use gpu')
parser.add_argument("--tune_wordemb", action="store_const", const=True, default=True)
parser.add_argument('--random_seed', type=int, default=42, help='0 if randomly initialize the model, other if fix the seed')
parser.add_argument('--broadcast', action="store_true", help='the parameter for infoNCE')
parser.add_argument('--gradient_accumulation_step', type=float, default=1)
parser.add_argument('--extra_step', action="store_true", help='(for test) step twice')
parser.add_argument('--save_checkpoint_all', action="store_true")
parser.add_argument('--save_checkpoint_best', action="store_true")
parser.add_argument('--temperature', type=float, default=1.0)
parser.add_argument('--checkpoint_step', type=float, default=10, help='Which steps output the training loss')
parser.add_argument('--use_eval', action="store_true")
parser.add_argument('--online_neg_num', type=int, default=0)
parser.add_argument('--add_save_params', action="store_true")
parser.add_argument("--train_ratio", type=float, default=1.0)
parser.add_argument('--bert_dir', type=str, default="bert-base-chinese")

# knn params
parser.add_argument('--alpha', type=float, default=1.0)
parser.add_argument('--beta1', type=float, default=1.0)
parser.add_argument('--beta2', type=float, default=1.0)
parser.add_argument('--knn', type=int, default=4)

args = parser.parse_args()
# command = ' '.join(['python'] + sys.argv)
# args.command = command