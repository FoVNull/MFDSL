import math
import pickle
import pandas as pd
import tensorflow as tf
import tensorflow_transform as tft
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


def tf2w_dic_build(file: str, others: list):
    word_dic = {}
    tf = {}
    word_count = 0
    tf_value = 0
    with open(file, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            for w in line.strip().split(" "):
                word_count += 1
                word_dic[w] = word_dic.get(w, 0)+1
    for k, v in word_dic.items():
        tf[k] = (v*10000)/word_count
        tf_value += tf[k]
    tf_value /= word_count

    word_dic.clear()
    word_count = 0
    tf2w = {}
    tf_other = {}
    for other_file in others:
        with open(other_file, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                for w in line.strip().split(" "):
                    word_count += 1
                    word_dic[w] = word_dic.get(w, 0) + 1
    for k, v in word_dic.items():
        tf_other[k] = math.log(v*10000/word_count, 1-tf_value)

    for k in tf.keys():
        if k in tf_other.keys():
            tf2w[k] = tf[k] * tf_other[k]
        else:
            tf2w[k] = tf[k] * math.log(0.5*10000/word_count, 1-tf_value)

    pickle.dump(sorted(tf2w.items(), key=lambda x: x[1], reverse=True), open("./reference/tf2w.pkl", 'wb'))


def tf2w_calculate():
    tf2w_dic = pickle.load(open("./reference/tf2w.pkl", 'rb'))
    print(tf2w_dic[0:10])


def tf_idf(file: str):
    sentences = []
    vocab = set()
    max_len = 0
    with open(file, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            vocab_list = line.strip().split(" ")
            max_len = max(max_len, len(vocab_list))
            sentences.append(line.strip())
            for v in vocab_list:
                vocab.add(v)
    # tokenizer = Tokenizer(num_words=len(vocab), oov_token="<OOV>")
    # tokenizer.fit_on_texts(sentences)  # 构建字典
    #
    # seq = tokenizer.texts_to_sequences(sentences)
    # print(type(seq))
    #
    # padded = pad_sequences(seq, padding="post", maxlen=5, truncating="post")
    # print(padded, type(padded))

    tokens = tf.compat.v1.string_split(sentences)
    print(tokens)
    indices = tft.compute_and_apply_vocabulary(tokens, top_k=len(vocab))

    bow_indices, weight = tft.tfidf(indices, len(vocab) + 1)

    print(bow_indices, weight)


def seed_select(dimension: int):
    senti_dic = {}
    with open("./reference/汉语情感词极值表.txt", 'r', encoding='gbk') as f:
        for line in f.readlines():
            w, v = line.strip().split("\t")
            senti_dic[w] = float(v)*5

    df = pd.read_excel("./reference/情感词汇本体.xlsx", header=0, keep_default_na=False)
    for i in range(len(df)):
        emotion_type = df.iloc[i]['情感分类']
        strength = df.loc[i, '强度']
        word = df.loc[i, '词语']
        if emotion_type == 'PC':
            continue
        if emotion_type[0] == 'P':
            senti_dic[word] = float(senti_dic.get(word, 0.0)) + strength
        if emotion_type[0] == 'N':
            senti_dic[word] = float(senti_dic.get(word, 0.0)) - strength

    tf2w = pickle.load(open("./reference/tf2w.pkl", 'rb'))

    p_seed_dic = {}
    n_seed_dic = {}

    for t in tf2w[:10000]:
        if t[0] not in senti_dic.keys():
            continue
        s = float(senti_dic[t[0]]) * float(t[1])
        if s > 0:
            p_seed_dic[t[0]] = s
        if s < 0:
            n_seed_dic[t[0]] = s
    p_seeds = sorted(p_seed_dic.items(), key=lambda x: x[1], reverse=True)[:dimension]
    n_seeds = sorted(n_seed_dic.items(), key=lambda x: x[1], reverse=False)[:dimension]

    # p_seeds = []
    # n_seeds = []
    # for tp in p_sorted:
    #     p_seeds.append(
    #         (tp[0], float(senti_dic[tp[0]]))
    #     )
    # for tp in n_sorted:
    #     n_seeds.append(
    #         (tp[0], float(senti_dic[tp[0]]))
    #     )

    with open("./reference/seeds.tsv", 'w', encoding='utf-8') as f:
        for tp in p_seeds:
            f.write(tp[0]+"\t"+str(tp[1])+"\n")
        for tp in n_seeds:
            f.write(tp[0]+"\t"+str(tp[1])+"\n")


tf_idf("./corpus/hotel/train_cut.txt")