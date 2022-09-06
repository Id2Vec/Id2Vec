import json
import os


def name_ref_count(path):

    with open(path, 'r') as f:
        res = json.loads(f.readline())
    print(len(res))
    for top_k in [1, 5, 10, 20, 50, 100]:
        cnt = 0
        for i in res:
            if i <= top_k:
                cnt += 1
        print('top ' + str(top_k) + ': ' + str(cnt / len(res)))

    return 0


if __name__ == '__main__':
    files = os.listdir('./dataset')
    for file in files:
        fp = os.path.join('./results', file)
        if file.startswith('name_ref_id2vec'):
            print(file)
            name_ref_count(fp)
