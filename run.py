import argparse
import pickle
from tqdm import tqdm
import numpy as np
from gensim.models import Word2Vec
from gensim.models import FastText

from utils import *
# from sopmi import SoPmi
# from validation.wc_validation import Validation
from scipy.spatial.distance import pdist

import sys
prefix = sys.path[0]

# class SimCalculator(Validation):
#     def __init__(self):
#         self.senti_vector = {}
#         with open("D:/python/mylibs/sgns.sogou.word", 'r') as f:
#             f.readline()
#             while True:
#                 _line = f.readline()
#                 if not _line:
#                     break
#                 _v = _line.strip().split(" ")
#                 self.senti_vector[_v[0]] = np.array(_v[1:301], dtype='float64')
#
#     def senti_sim(self, word1, word2):
#         return self.cosine_similarity(self.senti_vector[word1], self.senti_vector[word2])


if __name__ == '__main__':
    parse = argparse.ArgumentParser(description="build sentiment dictionary")
    parse.add_argument("--corpus", help="corpus path to train senti seeds")
    parse.add_argument("--weight", default=False, help="re-calculate word weight? [True/False]")
    parse.add_argument("--weight_schema", type=str, default='tf_idf',
                       help="the way to calculate word weight[mcw, tf_idf, mix]")
    parse.add_argument("--select_seeds", default=False, help="re-select seeds? [True/False]")
    parse.add_argument("--dimension", default=40, type=int,
                       help="dimension of seeds, [--dimension] positive seeds and [--dimension] negative seeds")
    parse.add_argument("--model", type=str, default='fasttext', help="[word2vec, fasttext]")
    parse.add_argument("--language", dest="lan", default="zh", help="choose from [zh, en]")
    args = parse.parse_args()

    irre_c = [f"{prefix}/corpus/smp_cut.txt"]
    if args.lan == "en":
        irre_c = [f"{prefix}/corpus/NYT_comment.txt"]

    # mcw 多语料调整权重
    assert args.weight_schema in ['mcw', 'tf_idf', 'mix', 'none'], \
        'you can choose: [mcw, tf_idf, mix, none]'
    # 调整数值，控制数值量刚便于进行分类
    sv_weight = 0
    if args.weight_schema == 'mcw':
        if args.weight == "True":
            mcw_dic_build(args.corpus, irre_c)
    if args.weight_schema == 'tf_idf':
<<<<<<< Updated upstream
        sv_weight = 0.1
=======
        sv_weight = 0.01
>>>>>>> Stashed changes
        if args.weight == "True":
            tf_idf_build(args.corpus)
    if args.weight_schema == 'mix':
        sv_weight = 0.01
        if args.weight == "True":
            mcw_dic_build(args.corpus, irre_c)
            tf_idf_build(args.corpus)
            mix_tf_build()
    if args.weight_schema == 'none':
        sv_weight = 0.01
        weight = seed_select(args.dimension, args.weight_schema, args.lan)
    else:
        weight = pickle.load(open(f"{prefix}/reference/output/"+args.weight_schema+".pkl", 'rb'))
        seed_select(args.dimension, args.weight_schema, args.lan)

    seeds = []
    seeds_weight = []
    with open(f"{prefix}/reference/output/seeds.tsv", 'r', encoding='utf-8') as f:
        for line in f.readlines():
            w, v = line.strip().split("\t")
            seeds.append(w)
            seeds_weight.append(float(v))

    sv_dic = {}

    assert args.model in ['word2vec', 'fasttext'], 'you can choose: [word2vec, fasttext]'
    if args.model == 'word2vec':
        model = Word2Vec.load(f"{prefix}/reference/wc_model/output")
    if args.model == 'fasttext':
        model = FastText.load(f"{prefix}/reference/wc_model/output")
    
    word_vector = pickle.load(open("./reference/output/wv.pkl", 'rb'))

    # sopmier = SoPmi("./corpus/hotel/all_cut.tsv", "./reference/output/seeds.tsv")
    # sim_calculator = SimCalculator()

    for tp in tqdm(weight):
        if tp[0] == '':
            continue
        sv_dic[tp[0]] = []
        for i in range(len(seeds)):
            # x, y = word_vector[seeds[i]], word_vector[tp[0]]
            # X=np.vstack([x,y])
            # similarity = 1-pdist(X,'cosine')[0]
            similarity = model.wv.similarity(seeds[i], tp[0])
            value = similarity * float(seeds_weight[i]) * sv_weight
            sv_dic[tp[0]].append(value)
        # sv_dic[tp[0]] = [similarity
        #                  * float(seeds_weight[i]) * sv_weight
        #                  for i in range(len(seeds))]
    pickle.dump(sv_dic, open(f"{prefix}/reference/output/sv.pkl", 'wb'))
