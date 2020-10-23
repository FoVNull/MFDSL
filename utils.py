import math
import pickle
from gensim.models import Word2Vec


def tf2w_dic_build(file: str, others: list):
    word_dic = {}
    tf = {}
    word_count = 0
    with open(file, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            for w in line.strip().split(" "):
                word_count += 1
                word_dic[w] = word_dic.get(w, 0)+1
    for k, v in word_dic.items():
        tf[k] = (v*10000)/word_count

    word_dic.clear()
    tf2w = {}
    tf_other = {}
    for other_file in others:
        with open(other_file, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                for w in line.strip().split(" "):
                    word_count += 1
                    word_dic[w] = word_dic.get(w, 0) + 1
    for k, v in word_dic.items():
        tf_other[k] = math.log((v+tf.get(k, 0))*10000/word_count, 0.5)

    for k in tf.keys():
        if k in tf_other.keys():
            tf2w[k] = tf[k] * tf_other[k]
        else:
            tf2w[k] = tf[k]
    # res = sorted(tf_idf.items(), key=lambda x: x[1], reverse=True)
    # print(res[0:20])

    pickle.dump(tf2w, open("reference/tf2w.pkl", 'wb'))


def tf2w_calculate(words: list) -> list:
    tf2w_dic = pickle.load(open("reference/tf2w.pkl", 'rb'))
    return [tf2w_dic[word] for word in words]


def cos_similarity(word1, word2) -> float:
    model = Word2Vec.load("reference/output")
    return model.wv.similarity(word1, word2)

def seed_select():
    print(0)


tf2w_dic_build("wordvector/corpus/smp_cut.txt", ["wordvector/corpus/test_corpora_cut.txt"])
# tf2w_list = tf2w_calculate(["肺炎", "不是"])
# print(tf2w_list)

