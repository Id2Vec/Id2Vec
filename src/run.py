# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Running scripts for Pretraining-textcnn
"""

from __future__ import absolute_import, division, print_function

import argparse

import numpy as np
import torch
import json
from src.pipeline import evaluate_pretrained_model, pretrain, evaluate_task_1_idbench, evaluate_task_2_rename

from utils.utils import logger
from utils.utils import set_seed
from src.data import TextDataset
from src.model import Encoder, TextCNN_1, TextCNN_2
from transformers import (WEIGHTS_NAME, AdamW, RobertaForMaskedLM, get_linear_schedule_with_warmup,
                          RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer)


def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--train_data_file", default=None, type=str, required=True,
                        help="The input training data file (a text file).")
    parser.add_argument("--exp_name", default=None, type=str, required=True,
                        help="Experiment name to be shown on wandb")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--textcnn_path", default=None, type=str, required=False,
                        help="The path for eval textcnn model.")

    ## Other parameters
    parser.add_argument("--eval_data_file", default=None, type=str,
                        help="An optional input evaluation data file to evaluate the perplexity on (a text file).")
    parser.add_argument("--test_data_file", default=None, type=str,
                        help="An optional input evaluation data file to evaluate the perplexity on (a text file).")
    parser.add_argument("--eval_idbench_data_file", default=None, type=str,
                        help="An optional input evaluation data file to evaluate the perplexity on (a text file).")
    parser.add_argument("--eval_rename_data_file", default=None, type=str,
                        help="An optional input evaluation data file to evaluate the perplexity on (a text file).")

    parser.add_argument("--model_name_or_path", default=None, type=str,
                        help="The model checkpoint for weights initialization.")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Optional pretrained tokenizer name or path if not the same as model_name_or_path")
    parser.add_argument("--block_size", default=-1, type=int,
                        help="Optional input sequence length after tokenization.")
    parser.add_argument("--var_size", default=32, type=int,
                        help="Token length for variable, after tokenization.")

    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_test", action='store_true',
                        help="Whether to run eval on the dev set.")

    parser.add_argument("--do_freeze_encoder", action='store_true',
                        help="Whether to freeze encoder.")
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")

    parser.add_argument("--train_batch_size", default=4, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--eval_batch_size", default=4, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument("--eval_interval", default=7000, type=int,
                        help="The number of steps between each evaluation run.")
    parser.add_argument("--cluster_weight", default=1.0, type=float,
                        help="weight of cluster loss.")

    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument('--num_train_epochs', type=int, default=42,
                        help="num_train_epochs")

    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--gpu', action='store_true',
                        help="whether to use GPU if available")

    # args for textcnn
    parser.add_argument('--textcnn-type', type=str, default='1', help='which model arch to choose')
    parser.add_argument('-dropout', type=float, default=0.5, help='dropout probability [default: 0.5]')
    parser.add_argument('-embed-dim', type=int, default=768, help='number of embedding dimension [default: 128]')
    parser.add_argument('-kernel-num', type=int, default=10, help='number of kernels')
    parser.add_argument('-kernel-sizes', type=str, default='3,4,5',
                        help='comma-separated kernel size to use for convolution')

    args = parser.parse_args()

    # Set up device	
    device = torch.device("cuda" if (torch.cuda.is_available() and args.gpu) else "cpu")
    args.n_gpu = torch.cuda.device_count()
    args.device = device

    # Setup logging
    logger.warning("device: %s, n_gpu: %s", device, args.n_gpu)

    # Set seed
    set_seed(args.seed)

    # Init model
    config = RobertaConfig.from_pretrained(args.model_name_or_path)
    tokenizer = RobertaTokenizer.from_pretrained(args.tokenizer_name, do_lower_case=args.do_lower_case)
    base_encoder = RobertaForMaskedLM.from_pretrained(args.model_name_or_path, config=config)
    encoder = Encoder(base_encoder, config, tokenizer, args)
    # encoder.load_state_dict(torch.load(args.encoder_path), strict=False)

    kernel_sizes = [int(k) for k in args.kernel_sizes.split(',')]
    if args.textcnn_type == '1':
        model = TextCNN_1(embed_dim=768, class_num=2, kernel_num=args.kernel_num, kernel_sizes=kernel_sizes, dropout=args.dropout)
    elif args.textcnn_type == '2':
        model = TextCNN_2(embed_dim=768, class_num=2, kernel_num=args.kernel_num, kernel_sizes=kernel_sizes, dropout=args.dropout)
    elif args.textcnn_type == '2':
        model = TextCNN_1(embed_dim=768, class_num=2, kernel_num=args.kernel_num, kernel_sizes=kernel_sizes, dropout=args.dropout)
    else:
        assert 0
    if args.do_freeze_encoder:
        # freeze encoder
        for param in encoder.parameters():
            param.requires_grad = False
    if args.do_eval:
        model.load_state_dict(torch.load(args.textcnn_path), strict=False)
    # multi-gpu training (should be after apex fp16 initialization)
    encoder.to(args.device)
    model.to(args.device)

    if args.n_gpu > 1:
        encoder = torch.nn.DataParallel(encoder)
        model = torch.nn.DataParallel(model)

    logger.info("Training/evaluation parameters %s", args)

    # Pre-training
    if args.do_train:
        train_dataset = TextDataset(tokenizer, args, args.train_data_file)
        pretrain(args, train_dataset, encoder, model, tokenizer, eval_interval=args.eval_interval)

    if args.do_eval:
        evaluate_task_1_idbench(args, encoder, model, tokenizer)
        # evaluate_task_2_rename(args, encoder, model, tokenizer)


if __name__ == "__main__":
    main()
