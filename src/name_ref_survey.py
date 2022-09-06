import json
import statistics


def name_ref_survey(path):
    print(path)
    result = []
    with open(path, 'r') as f:
        res = json.loads(f.readline())
    print(len(res))

    for i in res:
        var1 = i[1][0]
        var2 = i[1][1]
        for cand in i[2]:
            if var2 in cand:
                if var1 == 'core':
                    print(i[1], i[2])
                    print(i)
                    break

    print(len(res))
    return result


def levenshteinDistance(s1, s2):
    if len(s1) > len(s2):
        s1, s2 = s2, s1
    distances = range(len(s1) + 1)
    for i2, c2 in enumerate(s2):
        distances_ = [i2 + 1]
        for i1, c1 in enumerate(s1):
            if c1 == c2:
                distances_.append(distances[i1])
            else:
                distances_.append(1 + min((distances[i1], distances[i1 + 1], distances_[-1])))
        distances = distances_
    return distances[-1]


def calc_dist(path):
    print(path)
    result = []
    with open(path, 'r') as f:
        res = json.loads(f.readline())
    print(len(res))
    res = res[:400]
    for i in res:
        var1 = i[1][0]
        var2 = i[1][1]
        var3 = i[2][0][0]
        dist = levenshteinDistance(var1, var3)
        result.append(dist)

    print("Standard Deviation of the sample is % s " % (statistics.stdev(result)))
    print("Mean of the sample is % s " % (statistics.mean(result)))
    return


'''
name_ref_survey('./dataset/name_ref_survey_id2vec-textcnn_5_epoch_6')
# name_ref_survey('./dataset/name_ref_survey_id2vec-textcnn_5_epoch_6')
# calc_dist('./dataset/name_ref_survey_id2vec-textcnn_5_epoch_6')
# name_ref_survey('./dataset/name_ref_survey_id2vec-textcnn_js_2_epoch_4')
# calc_dist('./dataset/name_ref_survey_id2vec-textcnn_js_2_epoch_4')
'''

# name_ref_survey(path_to_result)
