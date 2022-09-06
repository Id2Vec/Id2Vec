import json
import os
import random


def load_data(root_path='./dataset'):
    res = []
    fp = os.path.join(root_path, 'context_large_pair_wise.jsonl')
    with open(fp, 'r') as f:
        lines = f.readlines()
    for row in lines:
        data = json.loads(row)
        res.append(data)
    return res


def convert_data_format(input_data=None):
    if input_data is None:
        input_data = load_data()
    res = []
    for row in input_data:
        var1 = row['var1']
        var2 = row['var2']
        context1 = row['context1']
        context2 = row['context2']
        for ct in context1:
            code = ct[0]
            mask_location = ct[1]
    return res


if __name__ == '__main__':
    dataset_path = './dataset'
    data = load_data(dataset_path)
    data = convert_data_format(data)
