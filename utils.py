import pickle
import pandas as pd
import tensorflow as tf
from senticnet.senticnet import SenticNet
from senticnet.babelsenticnet import BabelSenticNet


def mcw_dic_build(file: str, others: list):
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
        tf_value += v
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
        # tf_other[k] = math.log(v*10000/word_count, tf_value)
        tf_other[k] = tf_value/v

    for k in tf.keys():
        if k in tf_other.keys():
            tf2w[k] = tf[k] * tf_other[k]
        else:
            tf2w[k] = tf[k] * (tf_value/0.5)
                      #math.log(0.5*10000/word_count, tf_value)

    pickle.dump(sorted(tf2w.items(), key=lambda x: x[1], reverse=True), open("./reference/mcw.pkl", 'wb'))


def tf_idf_build(file: str):
    sentences = []
    vocab = set()
    max_len = 0
    with open(file, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            vocab_list = line.strip().split(" ")
            max_len = max(max_len, len(vocab_list))
            sentences.append(vocab_list)
            for v in vocab_list:
                vocab.add(v)
    tk = tf.keras.preprocessing.text.Tokenizer()
    tk.fit_on_texts(sentences)
    matrix = tk.sequences_to_matrix(tk.texts_to_sequences(sentences), mode='tfidf')

    tf_idf = {}
    for i in range(1, len(matrix[0])):
        tf_idf[tk.index_word[i]] = sum(matrix[:, i])

    pickle.dump(sorted(tf_idf.items(), key=lambda x: x[1], reverse=True), open("./reference/tf_idf.pkl", 'wb'))


def mix_tf_build():
    tf2w = pickle.load(open("./reference/mcw.pkl", 'rb'))
    tf_idf = pickle.load(open("./reference/tf_idf.pkl", 'rb'))

    mix_tf = {}
    for tp in tf2w:
        mix_tf[tp[0]] = mix_tf.get(tp[0], 1) * tp[1]

    for tp in tf_idf:
        mix_tf[tp[0]] = mix_tf.get(tp[0], 1) * tp[1]

    pickle.dump(sorted(mix_tf.items(), key=lambda x: x[1], reverse=True), open("./reference/mix.pkl", 'wb'))


def seed_select(dimension: int, weight_schema, language):
    sn = SenticNet()
    senti_dic = {}
    with open("./reference/BosonNLP_sentiment_score.txt", 'r', encoding='UTF-8') as f:
        for line in f.readlines():
            if len(line.strip().split(" ")) < 2:
                continue
            w, v = line.strip().split(" ")
            senti_dic[w] = float(v)

    assert language in ['zh', 'en'], "only support [zh, en]"

    if language == 'zh':
        bsn = BabelSenticNet('cn')
        for concept in bsn.data.keys():
            try:
                senti_dic[concept] = float(bsn.polarity_value(concept)) + senti_dic.get(concept, 0.0)
            except KeyError as e:
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
        for concept in sn.data.keys():
            try:
                senti_dic[concept] = float(sn.polarity_value(concept)) + senti_dic.get(concept, 0.0)
            except KeyError as e:
                print("unexpected problem! feedback:github.com/FoVNull")

    assert weight_schema in ['mcw', 'tf_idf', 'mix'], \
        'you can choose: [mcw, tf_idf, mix]'
    weight = pickle.load(open("./reference/"+weight_schema+".pkl", 'rb'))

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

    with open("./reference/seeds.tsv", 'w', encoding='utf-8') as f:
        for tp in p_seeds:
            f.write(tp[0]+"\t"+str(tp[1])+"\n")
        for tp in n_seeds:
            f.write(tp[0]+"\t"+str(tp[1])+"\n")