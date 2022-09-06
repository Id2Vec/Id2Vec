import json
import os
import random


def collect_identifiers(root_path):
    res_path = os.path.join(root_path, 'identifiers.jsonl')
    if os.path.exists(res_path):
        with open(res_path, 'r') as f:
            res = json.loads(f.readline())
        return res
    else:
        res = []
    files = os.listdir(root_path)
    for file in files:
        if not (file.endswith('.jsonl') and 1):
            continue
        fp = os.path.join(root_path, file)
        with open(fp, 'r') as f:
            lines = f.readlines()
        for row in lines:
            data = json.loads(row)
            token = data['before']
            if token not in res:
                res.append(token)
            token = data['after']
            if token not in res:
                res.append(token)
    print(len(res))
    with open(res_path, 'w') as f:
        f.write(json.dumps(res))
    return res


def convert_original_dataset_into_new_textcnn_model_dataset(root_path):
    identifiers = collect_identifiers(root_path)
    files = ['pt_train.jsonl', 'pt_valid.jsonl', 'ft_train.jsonl', 'ft_test.jsonl', 'ft_valid.jsonl']
    for file in files:
        rng = random.Random(1234)
        fp = os.path.join(root_path, file)
        op = os.path.join(root_path, 'new_' + file)
        res = []
        with open(fp, 'r') as f:
            lines = f.readlines()
        for row in lines:
            sample1, sample2 = '', ''
            tmp = {}
            data = json.loads(row)
            tmp['source'] = data['source']
            tmp['before'] = data['before']
            tmp['mask_locations'] = data['mask_locations']
            tmp['label'] = 1
            res.append(tmp)

            tmp = {}
            data = json.loads(row)
            tmp['source'] = data['source']
            tmp['before'] = data['after']
            tmp['mask_locations'] = data['mask_locations']
            tmp['label'] = 1
            res.append(tmp)

            while True:
                sample1 = rng.choice(identifiers)
                if sample1 != data['before'] and sample1 != data['after']:
                    break
            tmp = {}
            data = json.loads(row)
            tmp['source'] = data['source']
            tmp['before'] = sample1
            tmp['mask_locations'] = data['mask_locations']
            tmp['label'] = 0
            res.append(tmp)

            while True:
                sample2 = rng.choice(identifiers)
                if sample2 != data['before'] and sample2 != data['after'] and sample1 != sample2:
                    break
            tmp = {}
            data = json.loads(row)
            tmp['source'] = data['source']
            tmp['before'] = sample2
            tmp['mask_locations'] = data['mask_locations']
            tmp['label'] = 0
            res.append(tmp)

        with open(op, 'w') as f:
            for i in res:
                f.write(json.dumps(i) + '\n')
    pass


if __name__ == '__main__':
    dataset_path = './small-dataset'
    collect_identifiers(dataset_path)
    convert_original_dataset_into_new_textcnn_model_dataset(dataset_path)

