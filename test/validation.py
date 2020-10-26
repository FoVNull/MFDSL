import numpy as np
from gensim.models import Word2Vec
from gensim.models import FastText
import pickle


class Validation:
    def __init__(self):
        self.model = FastText.load("../reference/wc_model/output")
        self.senti_vector = pickle.load(open("../sv.pkl", 'rb'))

    @staticmethod
    def cosine_similarity(x, y):
        res = np.array([[x[i] * y[i], x[i] * x[i], y[i] * y[i]] for i in range(len(x))])
        cos = sum(res[:, 0]) / (np.sqrt(sum(res[:, 1])) * np.sqrt(sum(res[:, 2])))
        return cos

    def senti_sim(self, word1, word2):
        print("word vectors: ", self.model.wv.similarity(word1, word2))

        senti_cos = self.cosine_similarity(self.senti_vector[word1], self.senti_vector[word2])
        print("senti_vectors: ", senti_cos)

    def most_sim(self, word, number):
        print("word vectors: ", self.model.wv.most_similar(word, topn=number))
        senti_dic = {}
        for tp in self.senti_vector.items():
            senti_dic[tp[0]] = self.cosine_similarity(self.senti_vector[word], self.senti_vector[tp[0]])
        most_list = sorted(senti_dic.items(), key=lambda x: x[1], reverse=True)[10:20]
        print("senti_vectors: ", most_list)


validator = Validation()
print("肺炎", "愤怒")
validator.senti_sim("肺炎", "愤怒")

print("疫情", "愤怒")
validator.senti_sim("疫情", "愤怒")
print("疫情", "高兴")
validator.senti_sim("疫情", "高兴")

print("医生", "致敬")
validator.senti_sim("医生", "致敬")
# validator.most_sim("疫情", number=10)
