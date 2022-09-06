from transformers import (RobertaForMaskedLM, RobertaConfig, RobertaTokenizer)
import torch
import torch.nn as nn

from utils.utils import example_to_almost_features


def sample2tensor(input_ids, labels, identifier, mask_indices, m_label):
    """
    Get one training item
    NOTICE: only pass the len of the identifier!
    """
    # print(self.examples[i].m_label)
    # print(type(self.examples[i].m_label))
    return (
        torch.tensor(input_ids),
        torch.tensor(labels),
        torch.tensor(len(identifier)),
        torch.tensor(mask_indices),
        torch.tensor([m_label, 1 - m_label], dtype=torch.float),
    )


def compareTwoVar(args, var1, var2, context, maskLocations, tokenizer: RobertaTokenizer, model, encoder):

    pre = 0
    context1 = ''
    context2 = ''
    maskLocations1 = []
    maskLocations2 = []
    maskLocations.append((len(context), 0))

    for instance in maskLocations:
        begin = instance[0]
        end = instance[1]
        context1 += context[pre:begin]
        context2 += context[pre:begin]
        if (begin == len(context)):
            break
        maskLocations1.append((len(context1), len(context1) + len(var1)))
        maskLocations2.append((len(context2), len(context2) + len(var2)))
        context1 += var1
        context2 += var2
        pre = end
    var_ebd1, var_prob1 = distanceToMask(args, var1, context1, maskLocations1, tokenizer, model, encoder)
    var_ebd2, var_prob2 = distanceToMask(args, var2, context2, maskLocations2, tokenizer, model, encoder)
    # print(var_ebd2)
    cos = nn.CosineSimilarity(dim=0, eps=1e-8)
    cs = cos(var_ebd1, var_ebd2).item()
    # print(cs)
    probPair = torch.FloatTensor([1.0 * var_prob1 / (var_prob1 + var_prob2), var_prob2 / (var_prob1 + var_prob2)])
    return (100 * probPair[0].item(), 100 * probPair[1].item()), cs


def distanceToMask(args, var, context, maskLocations, tokenizer: RobertaTokenizer, model, encoder):

    input_ids, labels, identifier, _, mask_indices = example_to_almost_features(source=context, mask_locations=maskLocations,
                                                            identifier=var, tokenizer=tokenizer, args=args)
    data = sample2tensor(input_ids, labels, identifier, mask_indices, 1)
    input_ids = torch.unsqueeze(data[0], 0).to(args.device)
    labels = torch.unsqueeze(data[1], 0).to(args.device)
    mask_indices = torch.unsqueeze(data[3], 0).to(args.device)

    id_len = data[2].to(args.device)
    id_len = torch.unsqueeze(id_len, 0).to(args.device)

    hidden_states, embeddings, mean_embeddings = encoder(input_ids, id_len, labels, mask_indices)
    var_ebd, var_prob = model.prob(hidden_states, embeddings, 1)

    var_prob = var_prob.tolist()[0]
    return var_ebd, var_prob

