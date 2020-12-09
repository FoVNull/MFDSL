import pickle
import pandas as pd
import tensorflow as tf
from senticnet.senticnet import SenticNet
from senticnet.babelsenticnet import BabelSenticNet
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation


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

    pickle.dump(sorted(bi_tf.items(), key=lambda x: x[1], reverse=True), open("./reference/output/mcw.pkl", 'wb'))


def tf_idf_build(file: str):
    sentences = []
    doc_f = {}
    tf_idf = {}
    max_len = 0
    with open(file, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            vocab_list = line.strip().split("\t")[0].split(" ")
            max_len = max(max_len, len(vocab_list))
            sentences.append(vocab_list)
            for v in set(vocab_list):
                doc_f[v] = doc_f.get(v, 0) + 1
    tk = tf.keras.preprocessing.text.Tokenizer()
    tk.fit_on_texts(sentences)
    matrix = tk.sequences_to_matrix(tk.texts_to_sequences(sentences), mode='tfidf')

    for i in range(1, len(matrix[0])):
        tf_idf[tk.index_word[i]] = sum(matrix[:, i])

    # for vocab_list in tqdm(sentences):
    #     counter = collections.Counter(vocab_list)
    #     for i in counter.items():
    #         tf_ = i[1]/len(vocab_list)
    #         idf = math.log(len(sentences)/doc_f[i[0]])
    #         tf_idf[i[0]] = tf_idf.get(i[0], 0) + tf_ * idf

    pickle.dump(sorted(tf_idf.items(), key=lambda x: x[1], reverse=True), open("./reference/output/tf_idf.pkl", 'wb'))


def mix_tf_build():
    tf2w = pickle.load(open("./reference/output/mcw.pkl", 'rb'))
    tf_idf = pickle.load(open("./reference/output/tf_idf.pkl", 'rb'))

    mix_tf = {}
    for tp in tf2w:
        mix_tf[tp[0]] = mix_tf.get(tp[0], 1) * tp[1]

    for tp in tf_idf:
        mix_tf[tp[0]] = mix_tf.get(tp[0], 1) * tp[1]

    pickle.dump(sorted(mix_tf.items(), key=lambda x: x[1], reverse=True), open("./reference/output/mix.pkl", 'wb'))


def lda_build(file):
    seq = []
    with open(file, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            seq.append(line.strip().split("\t")[0])
    tf_vectorizer = CountVectorizer(max_df=1.0, min_df=0.1,
                                    max_features=500)
    tf = tf_vectorizer.fit_transform(seq)
    lda = LatentDirichletAllocation(n_components=1,
                                    max_iter=100,
                                    learning_method='batch')
    model = lda.fit(tf)
    tf_feature_names = tf_vectorizer.get_feature_names()

    print(lda.perplexity(tf))

    for topic_idx, topic in enumerate(model.components_):
        print("Topic #%d:" % topic_idx)
        print([(tf_feature_names[i], round(topic[i], 2))
               for i in topic.argsort()[20::-1]])


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
                senti_dic[word] = float(senti_dic.get(word, 0.0)) + strength / 10
            if emotion_type[0] == 'N':
                senti_dic[word] = float(senti_dic.get(word, 0.0)) - strength / 10

    if language == 'en':
        for concept in sn.data.keys():
            try:
                senti_dic[concept] = float(sn.polarity_value(concept)) + senti_dic.get(concept, 0.0)
            except KeyError:
                print("unexpected problem! feedback:github.com/FoVNull")

    assert weight_schema in ['mcw', 'tf_idf', 'mix'], \
        'you can choose: [mcw, tf_idf, mix]'
    weight = pickle.load(open("./reference/output/" + weight_schema + ".pkl", 'rb'))

    p_seed_dic = {}
    n_seed_dic = {}

    for t in weight:
        if t[0] not in senti_dic.keys():
            continue
        s = float(senti_dic[t[0]]) * float(t[1])
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

    with open("./reference/output/vocab.tsv", 'w', encoding='utf-8') as f:
        p_len = int(len(p_seed_dic) * 0.3)
        n_len = int(len(n_seed_dic) * 0.3)
        for tp in sorted(p_seed_dic.items(), key=lambda x: x[1], reverse=True)[:p_len]:
            f.write(tp[0] + "\t" + str(tp[1]) + "\n")
        for tp in sorted(n_seed_dic.items(), key=lambda x: x[1], reverse=False)[:n_len]:
            f.write(tp[0] + "\t" + str(tp[1]) + "\n")


# lda_build("./corpus/classics/classics_en_cut.txt")
