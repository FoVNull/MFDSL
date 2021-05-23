import pandas as pd
from validation.classify import Classifier
import argparse
import tensorflow as tf


class OneHot(Classifier):
    def __init__(self, args):
        super().__init__(args)

    def load_data(self):
        if args.base == "DUT":
            df = pd.read_excel("../reference/情感词汇本体.xlsx", header=0, keep_default_na=False)
            dut = []
            for i in range(len(df)):
                dut.append((df.loc[i, '词语'], df.iloc[i]['情感分类'], df.loc[i, '强度']))

            with open("../corpus/" + self.args.corpus, 'r', encoding='utf-8') as f:
                for line in f.readlines():
                    if len(line.strip().split("\t")) < 2:
                        continue
                    data = line.split("\t")[0].split(" ")
                    senti = line.strip().split("\t")[1]

                    vector = [0.0]*len(dut)
                    for w in data:
                        for i in range(len(dut)):
                            s = dut[i]
                            if w == s[0]:
                                if s[1][0] == 'P':
                                    vector[i] = float(s[2])
                                else:
                                    vector[i] = -float(s[2])

                    self.train_data.append((vector, senti))

        if args.base == "senticnet":
            sn_list = self.load_sentinet()
            self.build_train(sn_list)
        elif args.base == "socialsent":
            senti_dic = self.load_socialsent()
            self.build_train([(_[0], _[1]) for _ in senti_dic.items()])
        elif args.base == "mix":
            sn_list = self.load_mix()
            self.build_train(sn_list)
        else:
            sentences = []
            sentiment = []
            with open("../corpus/" + self.args.corpus, 'r', encoding='utf-8') as f:
                for line in f.readlines():
                    if len(line.strip().split("\t")) < 2:
                        continue
                    data = line.split("\t")[0].split(" ")
                    senti = line.strip().split("\t")[1]
                    sentiment.append(senti)
                    sentences.append(data)
            tk = tf.keras.preprocessing.text.Tokenizer()
            tk.fit_on_texts(sentences)
            matrix = tk.sequences_to_matrix(tk.texts_to_sequences(sentences), mode='binary')
            for _i, seq in enumerate(matrix):
                self.train_data.append((seq, sentiment[_i]))

    @staticmethod
    def load_sentinet():
        from senticnet.senticnet import SenticNet
        sn = SenticNet()
        sn_list = []
        for concept in sn.data.keys():
            sn_list.append((concept, float(sn.polarity_value(concept))))

        return sn_list

    @staticmethod
    def load_socialsent():
        senti_dic = {}
        with open("../reference/stf_adj_2000.tsv") as f:
            for line in f.readlines():
                try:
                    w, v = line.split("\t")[:2]
                except Exception:
                    print(line.split("\t"))
                senti_dic[w] = senti_dic.get(w, 0.0) + float(v)
        with open("../reference/stf_freq_2000.tsv") as f:
            for line in f.readlines():
                w, v = line.split("\t")[:2]
                senti_dic[w] = senti_dic.get(w, 0.0) + float(v)

        return senti_dic

    @staticmethod
    def load_mix():
        from senticnet.senticnet import SenticNet
        sn = SenticNet()
        senti_dic = OneHot.load_socialsent()
        for concept in sn.data.keys():
            if concept == "helpful":
                continue
            senti_dic[concept] = float(sn.polarity_value(concept)) + senti_dic.get(concept, 0.0)

        return [(_[0], _[1]) for _ in senti_dic.items()]

    def build_train(self, sn_list):
        with open("../corpus/" + self.args.corpus, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                if len(line.strip().split("\t")) < 2:
                    continue
                data = line.split("\t")[0].split(" ")
                senti = line.strip().split("\t")[1]
                vector = [0.0] * len(sn_list)
                for w in data:
                    for _i in range(len(sn_list)):
                        s = sn_list[_i]
                        if w == s[0]:
                            vector[_i] = float(s[1])
                self.train_data.append((vector, senti))


if __name__ == '__main__':
    parse = argparse.ArgumentParser(description="sentiment classify validation")
    parse.add_argument("--corpus", type=str, default="classics/classics_test.tsv", help="specify corpus")
    parse.add_argument("--random_count", dest="count", default=10)
    parse.add_argument("--base", type=str, default="mix", help="specific one-hot embedding method")

    args = parse.parse_args()

    classifier = OneHot(args)
    classifier.load_data()
    acc_sum = 0
    eva_sum = [[0, 0], [0, 0], [0, 0]]
    for seed in range(args.count):
        _eva, _acc = classifier.train(seed)
        for i in range(3):
            eva_sum[i][0] += _eva[i][0]
            eva_sum[i][1] += _eva[i][1]
        acc_sum += _acc
        print('#'*50)
    legend = ["precision", "recall", "f1"]
    for i, lg in enumerate(legend):
        value = [eva_sum[i][0]/args.count, eva_sum[i][1]/args.count]
        print(lg, value, sum(value)/2)
    print(acc_sum/args.count)