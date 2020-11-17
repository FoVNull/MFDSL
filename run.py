import argparse
import pickle
from tqdm import tqdm
from gensim.models import Word2Vec
from gensim.models import FastText

from utils import tf2w_dic_build, seed_select, tf2w_calculate, tf_idf_build, mix_tf_build

if __name__ == '__main__':
    parse = argparse.ArgumentParser(description="build sentiment dictionary")
    parse.add_argument("--weight", default=False, help="re-calculate word weight? [True/False]")
    parse.add_argument("--weight_schema", type=str, default='tf_idf',
                       help="the way to calculate word weight[tf2w, tf_idf, mix_tf]")
    parse.add_argument("--select_seeds", default=False, help="re-select seeds? [True/False]")
    parse.add_argument("--dimension", default=40, type=int,
                       help="dimension of seeds, [--dimension] positive seeds and [--dimension] negative seeds")
    parse.add_argument("--model", type=str, default='fasttext', help="[word2vec, fasttext]")
    parse.add_argument("--language", dest="lan", default="zh", help="choose from [zh, en]")
    args = parse.parse_args()

    assert args.weight_schema in ['tf2w', 'tf_idf', 'mix_tf'], \
        'you can choose: [tf2w, tf_idf, mix_tf]'
    if args.weight_schema == 'tf2w':
        if args.weight:
            tf2w_dic_build("./corpus/hotel/train_cut.txt",
                           ["./corpus/smp_cut.txt"]) # "./corpus/test_corpora_cut.txt",
    if args.weight_schema == 'tf_idf':
        if args.weight:
            tf_idf_build("./corpus/hotel/train_cut.txt")
    if args.weight_schema == 'mix_tf':
        if args.weight:
            tf2w_dic_build("./corpus/hotel/train_cut.txt",
                           ["./corpus/smp_cut.txt"]) # "./corpus/test_corpora_cut.txt",
            tf_idf_build("./corpus/hotel/train_cut.txt")
            mix_tf_build()

    weight = pickle.load(open("./reference/"+args.weight_schema+".pkl", 'rb'))

    if args.select_seeds:
        seed_select(args.dimension, args.weight_schema, args.len)

    seeds = []
    seeds_weight = []
    with open("./reference/seeds.tsv", 'r', encoding='utf-8') as f:
        for line in f.readlines():
            w, v = line.strip().split("\t")
            seeds.append(w)
            seeds_weight.append(float(v))

    sv_dic = {}
    assert args.model in ['word2vec', 'fasttext'], 'you can choose: [word2vec, fasttext]'
    if args.model == 'word2vec':
        model = Word2Vec.load("./reference/wc_model/output")
    if args.model == 'fasttext':
        model = FastText.load("./reference/wc_model/output")

    for tp in tqdm(weight):
        if tp[0] == '':
            continue
        sv_dic[tp[0]] = [model.wv.similarity(seeds[i], tp[0]) * float(seeds_weight[i])
                         for i in range(len(seeds))]

    pickle.dump(sv_dic, open("sv.pkl", 'wb'))
