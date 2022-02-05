import math

from bare_model import FeatureFusion_Model
from XLNet import XLNet_linear

import argparse
import pickle
import numpy as np
import jieba

import tensorflow as tf
from transformers import XLNetTokenizer, BertTokenizer, MPNetTokenizer
from kashgari.embeddings import WordEmbedding, TransformerEmbedding
from kashgari.tokenizers import BertTokenizer as K_BertTokenizer
from kashgari.tasks.classification import BiLSTM_Model
from kashgari_local import XLNetEmbedding, MPNetEmbedding, HFBertEmbedding
from sklearn.model_selection import train_test_split


def _xlnet_corpus_gen(path):
    with open(path, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            yield line.strip().split('\t')[0]


# def read_amazon(path, senti_dic):
#     x_data, y_data, features = [], [], []
#     with open(path + '/vali6000.tsv', 'r', encoding='utf-8') as f:
#         for line in f.readlines():
#             k, v = line.strip().split('\t')
#             line_features = np.array([])
#             # load x
#             x_data.append([word for word in k.split(" ") if word])
#             # load features
#             for word in k.split(" "):
#                 if word not in senti_dic.keys():
#                     continue
#                 if len(line_features) > 0:
#                     line_features += np.array(senti_dic[word])
#                 else:
#                     line_features = np.array(senti_dic[word])
#             # load y
#             y_data.append(v)
#             features.append(list(line_features / len(line_features)))
#     return x_data, y_data, features


def read_data(corpus_path, senti_dic):
    x_y_data = []
    with open(corpus_path, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            k, v = line.strip().split('\t')
            x_y_data.append((k, v))

    x_data, y_data, features = [], [], []
    # hotel: 810
    np.random.seed(114)
    np.random.shuffle(x_y_data)
    sv_dim = len(list(senti_dic.values())[0])
    for item in x_y_data:
        if not item[0]:
            continue
        x_data.append(item[0])
        y_data.append(item[1])
        line_features = []
        for word in jieba.cut(item[0], cut_all=True):
            # continue
            if word not in senti_dic.keys():
                line_features.append([0.] * sv_dim)
                continue
            if len(line_features) > 0:
                line_features.append(senti_dic[word])
            else:
                line_features.append(senti_dic[word])
        features.append(line_features)

    return x_data, y_data, features


class Trainer:
    def __init__(self, _args):
        self.model_type = _args.model_type
        if _args.model_type == "w2v":
            self.vocab_path = _args.model_folder + "/vocab.txt"
            self.vector_path = _args.model_folder + "/3C.vec"
        if _args.model_type == "bert":
            self.checkpoint_path = _args.model_folder + '/bert_model.ckpt'
            self.config_path = _args.model_folder + '/bert_config.json'
            self.vocab_path = _args.model_folder + '/vocab.txt'
        self.model_folder = _args.model_folder
        self.k = _args.k_fold
        self.output_name = _args.output_name

    def _embedding(self, **params):
        if self.model_type == "w2v":
            embedding = WordEmbedding(self.vector_path)
        elif self.model_type in ["xlnet", 'xlnet_zh']:
            # 输入语料用于构建xlnet词表，生成器形式
            embedding = XLNetEmbedding(self.model_folder, _xlnet_corpus_gen(params["xlnet_corpus"]))
        elif self.model_type == "mpnet":
            embedding = MPNetEmbedding(self.model_folder)
        else:
            embedding = HFBertEmbedding(self.model_folder)

        return embedding

    def _tokenizing(self, _x):
        sentences_tokenized = _x

        if self.model_type == 'bert':
            tokenizer = BertTokenizer(self.vocab_path)
            sentences_tokenized = [tokenizer.tokenize(seq) for seq in _x]
        if self.model_type == 'xlnet':
            tokenizer = XLNetTokenizer(self.model_folder + "/spiece.model")
            sentences_tokenized = [tokenizer.tokenize(" ".join(seq)) for seq in _x]
        if self.model_type == 'mpnet':
            tokenizer = MPNetTokenizer(self.model_folder + '/vocab.txt')
            sentences_tokenized = [tokenizer.tokenize(" ".join(seq)) for seq in _x]
        if self.model_type == 'xlnet_zh':
            tokenizer = XLNetTokenizer(self.model_folder + "/spiece.model")
            sentences_tokenized = [tokenizer.tokenize(seq) for seq in _x]

        return sentences_tokenized

    def train(self, corpus):
        x_data, y_data, features = read_data(corpus, pickle.load(open('./reference/output/sv.pkl', 'rb')))
        x_data = self._tokenizing(x_data)
        embedding = self._embedding(xlnet_corpus=corpus)

        callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)]

        data_gen = self.k_fold(self.k, x_data, y_data, features, part=2000)
        reports = []
        for x_test, y_test, test_features, \
            x_vali, y_vali, vali_features, \
            x_train, y_train, train_features in data_gen:
            model = FeatureFusion_Model(embedding, feature_D=len(train_features[0][0]))
            # model = BiLSTM_Model(embedding)
            # model = XLNet_linear(embedding, feature_D=len(train_features[0][0]))

            print("train-{}, test-{}".format(len(x_train), len(x_test)))
            with tf.device('/gpu:0'):
                model.fit(x_train=(x_train, train_features), y_train=y_train, 
                          x_validate=(x_vali, vali_features), y_validate=y_vali,
                          batch_size=32, epochs=100, callbacks=callbacks)
                reports.append(self.evaluate(model, x_test, y_test, test_features))

                # model.fit(x_train=x_train, y_train=y_train,
                #           batch_size=32, epochs=15, callbacks=None)
                # reports.append(self.evaluate_nonf(model, x_test, y_test))
        self.save_res(reports, self.k)

    @staticmethod
    def evaluate(model, x_test_pure, y_test, features):
        # y_test转置，便于model.evaluate处理多任务输出
        y_test = list(map(lambda x: list(x), zip(*y_test)))
        reports = model.evaluate((x_test_pure, features), y_test, batch_size=32)
        # for report in reports:
        #     print(report)
        # model.save("./models/output")
        return reports

    @staticmethod
    def evaluate_nonf(model, x_test_pure, y_test):
        reports = model.evaluate(x_test_pure, y_test, batch_size=32)
        return reports

    @staticmethod
    def k_fold(k, x, y, f, part=1000):
        step = math.ceil(len(x)/k)
        for i in range(0, len(x), step):
            yield x[i:i + part], y[i:i + part], f[i:i + part], \
                  x[i+part:i+2*part], y[i+part:i+2*part], f[i+part:i+2*part], \
                  x[:i] + x[i + 2*part:], y[:i] + y[i + 2*part:], f[:i] + f[i + 2*part:]

    def save_res(self, reports, k):
        with open(f"./reports/{self.output_name}.tsv", 'w', encoding='utf-8') as f:
            acc, p, r, f1, auc = [], [], [], [], []
            for report in reports:
                acc.append(report[0]['detail']['accuracy'])
                p.append(report[0]['detail']['macro avg']['precision'])
                r.append(report[0]['detail']['macro avg']['recall'])
                f1.append(report[0]['detail']['macro avg']['f1-score'])
                auc.append(report[0]['auc'])
            res = "task{};{}-Fold\n" \
                  "avg-macro:acc={}; p={}; r={}; f1={}; auc{}\n" \
                  "max-macro:acc={}; p={}; r={}; f1={}; auc{}\n" \
                  "min-macro:acc={}; p={}; r={}; f1={}; auc{}\n" \
                .format(0, k,
                        sum(acc) / k, sum(p) / k, sum(r) / k, sum(f1) / k, sum(auc) / k,
                        max(acc), max(p), max(r), max(f1), max(auc), min(acc), min(p), min(r), min(f1), min(auc))
            f.write(res)
            f.write(f'std:{np.std(f1)}')
        # pickle.dump((acc, p, r, f1, auc), open("./details.pkl", 'wb'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="training model")

    parser.add_argument("--model_folder", type=str, help="universal pretrained model path")
    parser.add_argument("--model_type", type=str, default='xlnet_zh',
                        help="transfer model type, [bert, w2v, xlnet, xlnet_zh, mpnet]")
    parser.add_argument("--k_fold", type=int, default=5, help="{k} fold validation")
    parser.add_argument("--output_name", type=str, help="reports file name")

    args = parser.parse_args()
    trainer = Trainer(args)

    trainer.train(corpus='./corpus/takeaway/all.tsv')
