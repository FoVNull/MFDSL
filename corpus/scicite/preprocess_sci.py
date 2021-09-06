import en_core_web_lg
from tqdm import tqdm
import pickle
import numpy as np

parser = en_core_web_lg.load()
data = []


def preprocess():
    word_flag = ['ADJ', 'ADV', 'INTJ', 'VERB']
    with open('raw.tsv', 'r', encoding='utf-8') as f:
        for line in tqdm(f.readlines()):
            arr = line.split('\t')
            text = arr[1]
            seq = ' '.join([token.text for token in parser(text) if token.pos_ in word_flag])
            data.append([seq, arr[2]])

    with open('all.txt', 'w', encoding='utf-8') as f:
        for e in data:
            f.write(e[0]+'\n')

    with open('all_cut.tsv', 'w', encoding='utf-8') as f:
        for e in data:
            f.write('\t'.join(e))


def gen_features():
    lexicon = pickle.load(open('../../reference/output/sv.pkl', 'rb'))
    dim = 0
    for i in lexicon.values():
        dim = len(i)
        break
    output = {}
    with open('raw.tsv', 'r', encoding='utf-8') as f:
        for line in tqdm(f.readlines()):
            _data = line.split('\t')
            seq = _data[1].split(' ')
            arr = np.zeros(dim)
            valid_words_num = 0
            for word in seq:
                if word not in lexicon.keys():
                    continue
                valid_words_num += 1
                arr += np.array(lexicon[word])
            output[int(_data[0])] = list(arr / valid_words_num)
    pickle.dump(output, open('./senti_features.pkl', 'wb'))


preprocess()
# gen_features()
