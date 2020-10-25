import argparse
import pickle
import pandas as pd
from pandas import DataFrame
from tqdm import tqdm
from gensim.models import Word2Vec

from utils import tf2w_dic_build, seed_select, tf2w_calculate

if __name__ == '__main__':
    parse = argparse.ArgumentParser(description="build sentiment dictionary")
    parse.add_argument("--tf2w", default=False, help="re-calculate tf2w? [True/False]")
    parse.add_argument("--select_seeds", default=False, help="re-select seeds? [True/False]")
    parse.add_argument("--dimension", default=40, type=int,
                       help="dimension of seeds, [--dimension] positive seeds and [--dimension] negative seeds")
    args = parse.parse_args()

    if args.tf2w:
        tf2w_dic_build("./corpus/smp_cut.txt", ["./corpus/test_corpora_cut.txt"])

    if args.select_seeds:
        seed_select(args.dimension)

    seeds = []
    seeds_weight = []
    with open("./reference/seeds.tsv", 'r', encoding='utf-8') as f:
        for line in f.readlines():
            w, v = line.strip().split("\t")
            seeds.append(w)
            seeds_weight.append(float(v)/10)

    tf2w = pickle.load(open("./reference/tf2w.pkl", 'rb'))[:50000]

    sv_dic = {}
    model = Word2Vec.load("./reference/wc_model/output")

    for tp in tqdm(tf2w):
        tf2w_value = float(tp[1])
        sv_dic[tp[0]] = [model.wv.similarity(seeds[i], tp[0]) * tf2w_value * float(seeds_weight[i])
                         for i in range(len(seeds))]

    pickle.dump(sv_dic, open("sv.pkl", 'wb'))
    # sv_matrix = DataFrame(sv_dic, index=seeds)
    # sv_matrix.to_csv(open("sv.csv", 'wb', encoding='utf-8'), index=False)
    # print(sv_matrix, sv_matrix.shape)
