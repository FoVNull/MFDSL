import pickle
import pandas as pd
import tensorflow as tf
from senticnet.senticnet import SenticNet
from senticnet.babelsenticnet import BabelSenticNet
import math
from tqdm import tqdm
import collections
import numpy as np


def mcw_dic_build(file: str, others: list):
    word_dic = {}
    tf_dic = {}
    word_count = 0
    tf_sum = 0
    with open(file, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            for w in line.strip().split("\t")[0].split(" "):
                word_count += 1
                word_dic[w] = word_dic.get(w, 0) + 1
    for k, v in word_dic.items():
        tf_dic[k] = (v * 10000) / word_count
        tf_sum += v
    tf_avg = ((tf_sum / len(word_dic)) * 10000) / word_count

    word_dic.clear()
    word_count = 0
    bi_tf = {}
    tf_other = {}
    for other_file in others:
        with open(other_file, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                for w in line.strip().split("\t")[0].split(" "):
                    word_count += 1
                    word_dic[w] = word_dic.get(w, 0) + 1
    for k, v in word_dic.items():
        tf_other[k] = tf_avg / v
        # tf_other[k] = math.log(v*1000/word_count, tf_avg)

    for k in tf_dic.keys():
        if k in tf_other.keys():
            bi_tf[k] = tf_dic[k] * tf_other[k]
        else:
            bi_tf[k] = tf_dic[k] * (tf_avg / 0.5)
            # bi_tf[k] = math.log(0.5*1000/word_count, tf_avg)

    # sum_exp = sum([math.exp(v) for v in bi_tf.values()])
    # for k, v in bi_tf.items():
    #     bi_tf[k] = math.exp(v)/sum_exp

    pickle.dump(sorted(bi_tf.items(), key=lambda x: x[1], reverse=True), open("./reference/output/mcw.pkl", 'wb'))


def tf_idf_build(file: str):
    sentences = []
    doc_f = {}
    tf_idf = {}
    max_len = 0
    _lexicon = set()
    with open(file, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            vocab_list = line.strip().split("\t")[0].split(" ")
            for w in vocab_list:
                _lexicon.add(w)
            max_len = max(max_len, len(vocab_list))
            sentences.append(vocab_list)
            for v in set(vocab_list):
                doc_f[v] = doc_f.get(v, 0) + 1
    # 对照方法 常规tf-idf算法
    # for vocab_list in tqdm(sentences):
    #     counter = collections.Counter(vocab_list)
    #     for i in counter.items():
    #         tf_ = i[1]/len(vocab_list)
    #         idf = math.log(len(sentences)/doc_f[i[0]])
    #         tf_idf[i[0]] = tf_idf.get(i[0], 0) + tf_ * idf

    # tf-igm
    # f_list = {}
    # for vocab_list in sentences:
    #     counter = collections.Counter(vocab_list)
    #     for v in _lexicon:
    #         if v in f_list.keys():
    #             f_list[v].append(counter[v]/len(vocab_list))
    #         else:
    #             f_list[v] = [counter.get(v, 0)]
    # igm = {}
    # for k, v in f_list.items():
    #     gravity_arg = list(reversed(np.argsort(v)))
    #     sum_ = sum([v[g] * (i+1) for i, g in enumerate(gravity_arg)])
    #     igm[k] = v[gravity_arg[0]] / sum_
    #
    # for vocab_list in tqdm(sentences):
    #     counter = collections.Counter(vocab_list)
    #     for i in counter.items():
    #         tf_ = i[1]/len(vocab_list)
    #         tf_idf[i[0]] = tf_idf.get(i[0], 0) + tf_ * igm[i[0]]

    tk = tf.keras.preprocessing.text.Tokenizer()
    tk.fit_on_texts(sentences)
    matrix = tk.sequences_to_matrix(tk.texts_to_sequences(sentences), mode='tfidf')

    # 每一行都过一层softmax
    for i in tqdm(range(len(matrix))):
        sum_exp = math.exp(sum(matrix[i]))
        for j in range(len(matrix[0])):
            if matrix[i][j] == 0:
                continue
            matrix[i][j] = math.exp(matrix[i][j])/sum_exp

    for i in range(1, len(matrix[0])):
        _value = sum(matrix[:, i])
        tf_idf[tk.index_word[i]] = _value

    pickle.dump(sorted(tf_idf.items(), key=lambda x: x[1], reverse=True), open("./reference/output/tf_idf.pkl", 'wb'))


def mix_tf_build():
    tf2w = pickle.load(open("./reference/output/mcw.pkl", 'rb'))
    tf_idf = pickle.load(open("./reference/output/tf_idf.pkl", 'rb'))

    mix_tf = {}
    for tp in tf2w:
        mix_tf[tp[0]] = mix_tf.get(tp[0], 1) * tp[1]

    for tp in tf_idf:
        mix_tf[tp[0]] = mix_tf.get(tp[0], 1) * tp[1]

    # 加一层softmax
    # sum_exp = sum([math.exp(v) for v in mix_tf.values()])
    # for k, v in mix_tf.items():
    #     mix_tf[k] = math.exp(v)/sum_exp

    pickle.dump(sorted(mix_tf.items(), key=lambda x: x[1], reverse=True), open("./reference/output/mix.pkl", 'wb'))


def seed_select(dimension: int, weight_schema, language):
    sn = SenticNet()
    senti_dic = {}
    # with open("./reference/BosonNLP_sentiment_score.txt", 'r', encoding='UTF-8') as f:
    #     for line in f.readlines():
    #         if len(line.strip().split(" ")) < 2:
    #             continue
    #         w, v = line.strip().split(" ")
    #         senti_dic[w] = float(v)

    assert language in ['zh', 'en'], "only support [zh, en]"

    if language == 'zh':
        bsn = BabelSenticNet('cn')
        for concept in bsn.data.keys():
            try:
                senti_dic[concept] = float(bsn.polarity_value(concept)) + senti_dic.get(concept, 0.0)
            except KeyError:
                print("unexpected problem! feedback:github.com/FoVNull")

        df = pd.read_excel("./reference/情感词汇本体.xlsx", header=0, keep_default_na=False)
        for i in range(len(df)):
            emotion_type = df.iloc[i]['情感分类']
            strength = df.loc[i, '强度']
            word = df.loc[i, '词语']
            if emotion_type == 'PC':
                continue
            if emotion_type[0] == 'P':
                senti_dic[word] = float(senti_dic.get(word, 0.0)) + strength/10
            if emotion_type[0] == 'N':
                senti_dic[word] = float(senti_dic.get(word, 0.0)) - strength/10

    if language == 'en':
        # SenticNet
        # for concept in sn.data.keys():
        #     try:
        #         if concept == "helpful":
        #             continue
        #         senti_dic[concept] = float(sn.polarity_value(concept)) + senti_dic.get(concept, 0.0)
        #     except KeyError:
        #         print("unexpected problem! feedback:github.com/FoVNull")
        # SocialSent
        with open("./reference/stf_adj_2000.tsv") as f:
            for line in f.readlines():
                try:
                    w, v = line.split("\t")[:2]
                except Exception:
                    print(line.split("\t"))
                senti_dic[w] = senti_dic.get(w, 0.0) + float(v) + 1.0
        with open("./reference/stf_freq_2000.tsv") as f:
            for line in f.readlines():
                w, v = line.split("\t")[:2]
                senti_dic[w] = senti_dic.get(w, 0.0) + float(v) + 1.0

    assert weight_schema in ['mcw', 'tf_idf', 'mix'], \
        'you can choose: [mcw, tf_idf, mix]'
    weight = pickle.load(open("./reference/output/" + weight_schema + ".pkl", 'rb'))

    # 先行生成词典
    # p_senti_dic = []
    # n_senti_dic = []
    # weight_keys = [t[0] for t in weight]
    # for item in senti_dic.items():
    #     if item[0] not in weight_keys:
    #         continue
    #     if float(item[1]) > 0:
    #         p_senti_dic.append((item[0], item[1]))
    #     elif float(item[1]) < 0:
    #         n_senti_dic.append((item[0], item[1]))
    # with open("./reference/output/vocab.tsv", 'w', encoding='utf-8') as f:
    #     # 选取超过阈值情感词
    #     p_len = int(len(p_senti_dic) * 1)
    #     n_len = int(len(n_senti_dic) * 1)
    #     print(p_len, n_len)
    #     for tp in sorted(p_senti_dic, key=lambda x: x[1], reverse=True)[:p_len]:
    #         f.write(tp[0] + "\t" + str(tp[1]) + "\n")
    #     for tp in sorted(n_senti_dic, key=lambda x: x[1], reverse=False)[:n_len]:
    #         f.write(tp[0] + "\t" + str(tp[1]) + "\n")

    p_seed_dic = {}
    n_seed_dic = {}

    for t in weight:
        if t[0] not in senti_dic.keys():
            continue
        s = float(senti_dic[t[0]]) * float(t[1])
        # s = float(senti_dic[t[0]])
        if s > 0:
            p_seed_dic[t[0]] = s
        if s < 0:
            n_seed_dic[t[0]] = s
        if len(p_seed_dic) == dimension and len(n_seed_dic) == dimension:
            break
    p_seeds = sorted(p_seed_dic.items(), key=lambda x: x[1], reverse=True)[:dimension]
    n_seeds = sorted(n_seed_dic.items(), key=lambda x: x[1], reverse=False)[:dimension]

    with open("./reference/output/seeds.tsv", 'w', encoding='utf-8') as f:
        for tp in p_seeds:
            f.write(tp[0] + "\t" + str(tp[1]) + "\n")
        for tp in n_seeds:
            f.write(tp[0] + "\t" + str(tp[1]) + "\n")

    # with open("./reference/output/vocab_0.tsv", 'w', encoding='utf-8') as f:
    #     p_len = int(len(p_seed_dic) * 1)
    #     n_len = int(len(n_seed_dic) * 1)
    #     for tp in sorted(p_seed_dic.items(), key=lambda x: x[1], reverse=True)[:p_len]:
    #         f.write(tp[0] + "\t" + str(tp[1]) + "\n")
    #     for tp in sorted(n_seed_dic.items(), key=lambda x: x[1], reverse=False)[:n_len]:
    #         f.write(tp[0] + "\t" + str(tp[1]) + "\n")
    # exit(99)
