# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from typing import List, Tuple
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, MSELoss
from transformers import RobertaForMaskedLM, RobertaTokenizer, RobertaConfig, RobertaModel
from transformers.modeling_outputs import MaskedLMOutput
import wandb
from utils.utils import logger
from torch import Tensor


class Encoder(RobertaModel):
    def __init__(self, encoder: RobertaForMaskedLM, config: RobertaConfig, tokenizer: RobertaTokenizer, args):
        super().__init__(config)
        self.encoder: RobertaForMaskedLM = encoder  # func: ("input_ids", "attention_mask") -> list
        self.config = config
        self.tokenizer: RobertaTokenizer = tokenizer
        self.args = args

    def forward(self, input_ids: Tensor, id_len: Tensor, labels: Tensor = None, o_mask_indices: Tensor = None) -> Tuple or Tensor:
        """
        During training and eval, labels will not be None
        During testing, labels will be None
        """

        # Firstly, get the output of the base class (RobertaForMaskedLM)
        # `logits` has the same shape as `input_ids`
        # not attending to the padding tokens (id=1)

        # print(input_ids)
        # print(id_len)
        # print(o_mask_indices)
        batch_size = input_ids.shape[0]
        mlm_output: MaskedLMOutput = self.encoder(
            input_ids,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
            labels=labels,
            return_dict=True,
            output_hidden_states=True
        )
        hidden_states = mlm_output.hidden_states[0]  # the final embeddings are in [0] (among 13)
        assert hidden_states.shape == (batch_size, self.args.block_size, self.config.hidden_size)

        # Extract the mean embedding of the identifier in each example
        embeddings: List[
            Tensor] = []  # each of which contains k embeddings of the identifier in the corresponding example
        mean_embeddings: List[Tensor] = []
        mask_indices_list = []

        for row in range(batch_size):
            num_pieces = id_len[row]
            mask_indices = (input_ids[row] == self.tokenizer.mask_token_id).nonzero(as_tuple=True)[0]  # we get the first (and only) element of the tuple
            # print('mask_indices 1: ', mask_indices)
            mask_indices = o_mask_indices[row][:id_len[row]]
            # print('mask_indices 2: ', mask_indices)
            # print('row of mask indices', mask_indices)
            assert mask_indices.dim() == 1
            mask_indices_list.append(mask_indices)

            # Get the embedding of the masked identifiers
            embeddings_this_eg = torch.reshape(
                hidden_states[row, mask_indices, :],
                (-1, num_pieces, self.config.hidden_size)
            )
            embeddings.append(embeddings_this_eg)
            mean_embeddings.append(embeddings_this_eg.mean(dim=0))
            assert mean_embeddings[-1].shape == (num_pieces, self.config.hidden_size)

        assert len(embeddings) == batch_size
        return hidden_states, embeddings, mean_embeddings


class TextCNN_1(nn.Module):
    def __init__(self, embed_dim, class_num, kernel_num, kernel_sizes, dropout=0.5):
        super(TextCNN_1, self).__init__()

        Ci = 1
        Co = kernel_num

        # arch 1
        self.convs1 = nn.ModuleList([nn.Conv2d(Ci, Co, (f, embed_dim), padding=(2, 0)) for f in kernel_sizes])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(Co * len(kernel_sizes), class_num)
        self.fc2 = nn.Linear(30, class_num)

        self.fc_dense_1 = nn.Linear(768, 300)
        self.fc_dense_2 = nn.Linear(300, 30)
        self.fc_dense_3 = nn.Linear(30, class_num)

        # ff
        self.sig = nn.Sigmoid()
        self.loss = nn.BCELoss()

    def forward(self, hidden_states, embeddings, m_label):
        # print(hidden_states.shape)
        # seq_embeddings = self.predict(hidden_states)

        seq_embeddings = hidden_states.mean(dim=1)
        # print(mean_seq_embeddings.shape)

        # print(seq_embeddings.shape)

        # assert 0
        # sub_token_ebd = torch.cat(embeddings, dim=0)
        # print('sub_token_ebd: ', sub_token_ebd.shape)
        identifier_ebd = []
        for row_sub_tokens_ebd in embeddings:
            # print('shape of sub tokens: ', row_sub_tokens_ebd.shape)
            row_sub_tokens_ebd = torch.reshape(row_sub_tokens_ebd, (1, -1, 768))
            # print('shape of sub tokens: ', row_sub_tokens_ebd.shape)
            row_identifier_ebd = self.predict(row_sub_tokens_ebd)
            # print(row_identifier_ebd.shape)
            identifier_ebd.append(row_identifier_ebd)
        # print(len(identifier_ebd))
        batched_identifiers_ebd = torch.cat(identifier_ebd, dim=0)
        # print('batched_identifiers_ebd: ', batched_identifiers_ebd.shape)
        seq_embeddings = self.fc_dense_1(seq_embeddings)
        seq_embeddings = self.fc_dense_2(seq_embeddings)
        out = torch.mul(seq_embeddings, batched_identifiers_ebd)
        # print(out.shape)
        out = self.fc_dense_3(out)
        # print(out.shape)
        out = self.sig(out)
        # print(out.shape)
        out = torch.reshape(out, (-1,))
        m_label = torch.reshape(m_label, (-1,))
        loss = self.loss(out, m_label)
        # print(loss)
        return loss

    def predict(self, x):
        x = x.unsqueeze(1)  # (N, Ci, token_num, embed_dim)
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs1]  # [(N, Co, token_num) * len(kernel_sizes)]
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  # [(N, Co) * len(kernel_sizes)]
        x = torch.cat(x, 1)  # (N, Co * len(kernel_sizes))
        x = self.dropout(x)
        # logit = self.fc(x)
        return x

    def prob(self, hidden_states, embeddings, m_label):
        seq_embeddings = self.predict(hidden_states)

        identifier_ebd = []
        for row_sub_tokens_ebd in embeddings:
            row_sub_tokens_ebd = torch.reshape(row_sub_tokens_ebd, (1, -1, 768))
            row_identifier_ebd = row_sub_tokens_ebd.mean(dim=1)
            row_identifier_ebd = self.fc_dense_1(row_identifier_ebd)
            row_identifier_ebd = self.fc_dense_2(row_identifier_ebd)
            identifier_ebd.append(row_identifier_ebd)

        batched_identifiers_ebd = torch.cat(identifier_ebd, dim=0)

        out = torch.mul(seq_embeddings, batched_identifiers_ebd)
        out = self.fc2(out)
        out = self.sig(out)
        # print(out)
        return batched_identifiers_ebd[0], out[0]


class TextCNN_2(nn.Module):
    def __init__(self, embed_dim, class_num, kernel_num, kernel_sizes, dropout=0.5):
        super(TextCNN_2, self).__init__()

        Ci = 1
        Co = kernel_num

        # arch 1
        self.convs1 = nn.ModuleList([nn.Conv2d(Ci, Co, (f, embed_dim), padding=(2, 0)) for f in kernel_sizes])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(Co * len(kernel_sizes), class_num)
        self.fc2 = nn.Linear(30, class_num)

        self.fc_dense_1 = nn.Linear(768, 300)
        self.fc_dense_2 = nn.Linear(768, 300)
        self.fc_dense_3 = nn.Linear(300, class_num)

        # ff
        self.sig = nn.Sigmoid()
        self.loss = nn.BCELoss()

    def forward(self, hidden_states, embeddings, m_label):
        seq_embeddings = self.predict(hidden_states)
        # print(seq_embeddings.shape)

        # sub_token_ebd = torch.cat(embeddings, dim=0)
        # print('sub_token_ebd: ', sub_token_ebd.shape)
        identifier_ebd = []
        for row_sub_tokens_ebd in embeddings:
            # print('shape of sub tokens: ', row_sub_tokens_ebd.shape)
            row_sub_tokens_ebd = torch.reshape(row_sub_tokens_ebd, (1, -1, 768))
            # print('shape of sub tokens: ', row_sub_tokens_ebd.shape)
            row_identifier_ebd = self.predict(row_sub_tokens_ebd)
            # print(row_identifier_ebd.shape)
            identifier_ebd.append(row_identifier_ebd)
        # print(len(identifier_ebd))
        batched_identifiers_ebd = torch.cat(identifier_ebd, dim=0)
        # print('batched_identifiers_ebd: ', batched_identifiers_ebd.shape)

        out = torch.mul(seq_embeddings, batched_identifiers_ebd)
        # print(out.shape)
        out = self.fc2(out)
        # print(out.shape)
        out = self.sig(out)
        # print(out.shape)
        out = torch.reshape(out, (-1,))
        m_label = torch.reshape(m_label, (-1,))
        loss = self.loss(out, m_label)
        # print(loss)
        return loss

    def predict(self, x):
        x = x.unsqueeze(1)  # (N, Ci, token_num, embed_dim)
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs1]  # [(N, Co, token_num) * len(kernel_sizes)]
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  # [(N, Co) * len(kernel_sizes)]
        x = torch.cat(x, 1)  # (N, Co * len(kernel_sizes))
        x = self.dropout(x)
        # logit = self.fc(x)
        return x

    def prob(self, hidden_states, embeddings, m_label):
        seq_embeddings = self.predict(hidden_states)

        identifier_ebd = []
        for row_sub_tokens_ebd in embeddings:
            row_sub_tokens_ebd = torch.reshape(row_sub_tokens_ebd, (1, -1, 768))
            row_identifier_ebd = self.predict(row_sub_tokens_ebd)
            identifier_ebd.append(row_identifier_ebd)
        batched_identifiers_ebd = torch.cat(identifier_ebd, dim=0)

        out = torch.mul(seq_embeddings, batched_identifiers_ebd)
        out = self.fc2(out)
        out = self.sig(out)
        # print(out)
        return batched_identifiers_ebd[0], out[0]
