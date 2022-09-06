import json
import os
import random
from typing import Dict, List, Tuple
import numpy as np
import torch
from torch.utils.data import Dataset
import logging
import re
from transformers import RobertaTokenizer

logger = logging.getLogger(__name__)  # WARNING: side effects
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)


def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYHTONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def example_to_almost_features(source: str, mask_locations: List[List[int]],
                               identifier: str, tokenizer: RobertaTokenizer, args) -> Tuple[List[int]]:
    """
    Generic function to convert source code to (????)
    Source code is preprocessed, tokenized and crop/pad into 512 tokens
    """

    def format(text):
        """
        Some preprocessing for the source code
        Author: Yuchen
        """
        text = re.sub(r"\b", r" ", text)  # insert a space into word boundaries
        text = re.sub(r"(\W)", r" \1 ", text)  # insert spaces around non-word characters
        text = re.sub(r"[ \t]+", r" ", text)  # shrink multiple spaces and tabs into one space
        text = text.strip()  # strip the leading and trailing spaces
        return text

    mask_locations += [[len(source)] * 2]  # add a dummy mask for easy loop later
    # print('mask_locations: ', mask_locations)
    input_ids = []
    labels = []
    done_mark = -1  # right before the undone part
    has_mask = False

    identifier_ids = tokenizer.encode(identifier, add_special_tokens=False, padding=False)
    # print('identifier_ids: ', identifier_ids)
    assert not (tokenizer.pad_token_id in identifier_ids)
    num_mask_pieces = len(identifier_ids)  # count the number of real tokens in ids_before

    mask_indices = []
    for i, loc in enumerate(mask_locations):
        unmasked_code = format(source[done_mark + 1: loc[0]])
        unmasked_ids = tokenizer.encode(unmasked_code, add_special_tokens=False)
        input_ids.extend(unmasked_ids)
        labels.extend(unmasked_ids)

        done_mark = loc[1] - 1

        # skip the dummy
        if i == len(mask_locations) - 1:
            break

        # only append if can do with the whole word
        if len(input_ids) + num_mask_pieces <= args.block_size - 2:
            for _ in range(num_mask_pieces):
                mask_indices.append(len(input_ids) + _ + 1)
            # input_ids.extend([tokenizer.mask_token_id] * num_mask_pieces)
            input_ids.extend(identifier_ids)
            labels.extend(identifier_ids)
            has_mask = True

        else:
            # exceeding the limit
            break

    if not has_mask:
        raise ValueError("There is no mask in the block_size limit")

    input_ids = [tokenizer.cls_token_id] + input_ids[:args.block_size - 2] + [tokenizer.sep_token_id]  # crop for tokens
    labels = [tokenizer.cls_token_id] + labels[:args.block_size - 2] + [tokenizer.sep_token_id]  # crop for tokens

    padding_length = args.block_size - len(input_ids)
    masks = [1] * len(input_ids) + [0] * padding_length  # mask of input_ids
    input_ids += [tokenizer.pad_token_id] * padding_length
    labels += [tokenizer.pad_token_id] * padding_length

    mask_padding_length = args.block_size - len(mask_indices)
    mask_indices = mask_indices + [0] * mask_padding_length

    return input_ids, labels, identifier_ids, masks, mask_indices
