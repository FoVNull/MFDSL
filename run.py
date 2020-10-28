import argparse
import pickle
from tqdm import tqdm
from gensim.models import Word2Vec
from gensim.models import FastText
from gensim.models import KeyedVectors

from utils import tf2w_dic_build, seed_select, tf2w_calculate

if __name__ == '__main__':
    parse = argparse.ArgumentParser(description="build sentiment dictionary")
    parse.add_argument("--tf2w", default=False, help="re-calculate tf2w? [True/False]")
    parse.add_argument("--select_seeds", default=False, help="re-select seeds? [True/False]")
    parse.add_argument("--dimension", default=40, type=int,
                       help="dimension of seeds, [--dimension] positive seeds and [--dimension] negative seeds")
    parse.add_argument("--model", type=str, default='fasttext', help="[word2vec, fasttext]")
    args = parse.parse_args()

    if args.tf2w:
        tf2w_dic_build("./corpus/hotel/train_cut.txt", ["./corpus/test.txt"])

    if args.select_seeds:
        seed_select(args.dimension)

    seeds = []
    seeds_weight = []
    with open("./reference/seeds.tsv", 'r', encoding='utf-8') as f:
        for line in f.readlines():
            w, v = line.strip().split("\t")
            seeds.append(w)
            seeds_weight.append(float(v)/10)

    tf2w = pickle.load(open("./reference/tf2w.pkl", 'rb'))

    sv_dic = {}
    assert args.model == 'fasttext' or args.model == 'word2vec', 'you can choose: [word2vec, fasttext]'
    if args.model == 'word2vec':
        model = Word2Vec.load("./reference/wc_model/output")
    if args.model == 'fasttext':
        model = FastText.load("./reference/wc_model/output")
        # model = KeyedVectors.load_word2vec_format('D:/python/mylibs/cc.zh.300.vec', binary=False)

    for tp in tqdm(tf2w):
        if tp[0] == '':
            continue
        tf2w_value = float(tp[1])
        sv_dic[tp[0]] = [model.wv.similarity(seeds[i], tp[0]) * float(seeds_weight[i])
                         for i in range(len(seeds))]

    pickle.dump(sv_dic, open("sv.pkl", 'wb'))
