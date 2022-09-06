from torch.utils.data import Dataset
import json
from typing import Dict, List, Tuple
import torch
from torch.utils.data import Dataset
from transformers import RobertaTokenizer
from utils.utils import logger, example_to_almost_features


class InputFeaturesForPretrain(object):
    """A single training/test features for an example."""

    def __init__(self,
                 input_ids: List[int],
                 labels: List[int],
                 identifier: List[int],
                 mask_indices: List[int],
                 m_label: int = None,
                 index: int = None
                 ):
        self.index = index
        self.input_ids = input_ids
        self.labels = labels
        self.identifier = identifier
        self.mask_indices = mask_indices
        self.m_label = m_label


def convert_example_to_features_for_pretrain(js: Dict, tokenizer: RobertaTokenizer, args):
    """
    Process ONE example
    Currently only for nameBefore.
    Return InputFeatures if success, None if fail.

    code -> source, words -> numbers (?)
    """
    m_label = int(js['label'])
    # print(m_label, type(m_label))
    input_ids, labels, identifier, _, mask_indices = example_to_almost_features(  # WARNING the return type may change
        js['source'], js['mask_locations'], js['before'], tokenizer, args)

    return InputFeaturesForPretrain(input_ids, labels, identifier, mask_indices, m_label)


class TextDataset(Dataset):
    def __init__(self, tokenizer: RobertaTokenizer, args, file_path=None):
        self.examples: List[InputFeaturesForPretrain] = []
        with open(file_path) as f:
            count = 0
            for line in f:
                count += 1
                js = json.loads(line.strip())
                # not append when there is no masks left, or the function return None
                try:
                    features = convert_example_to_features_for_pretrain(js, tokenizer, args)
                except ValueError as e:
                    # print('error ...', e)
                    continue
                features.index = len(self.examples)
                self.examples.append(features)
            logger.info(f"Got {len(self.examples)} training examples from {count}")

        if 'train' in file_path:
            for idx, example in enumerate(self.examples[:3]):
                logger.info("*** Example ***")
                remove_pad = lambda ids: ids if not (1 in ids) else ids[:ids.index(1)]
                logger.info("labels: {}".format(example.labels))
                input_tokens = tokenizer.convert_ids_to_tokens(remove_pad(example.input_ids))
                logger.info("input_tokens: {}".format([x.replace('\u0120', '_') for x in input_tokens]))
                logger.info("input_ids: {}".format(' '.join(map(str, example.input_ids))))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        """
        Get one training item
        NOTICE: only pass the len of the identifier!
        """
        # print(self.examples[i].m_label)
        # print(type(self.examples[i].m_label))
        return (
            torch.tensor(self.examples[i].input_ids),
            torch.tensor(self.examples[i].labels),
            torch.tensor(len(self.examples[i].identifier)),
            torch.tensor(self.examples[i].mask_indices),
            torch.tensor([self.examples[i].m_label, 1-self.examples[i].m_label], dtype=torch.float),
            torch.tensor(i)
        )
