import json
import jieba.posseg as pseg

stop_set = set(line.strip() for line in open("../reference/HIT_stopwords.txt", 'r', encoding='utf-8').readlines())


def cut_json(path):
    texts = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            dic = json.loads(line.strip())
            texts.append([word.word for word in pseg.cut(dic['content'], use_paddle=True)
                          if word.word not in stop_set and word.flag not in ['u', 'w', 'x', 'p', 'q', 'm', 'e', 'd']])
    save_data("../corpus/smp_cut.txt", texts)


def cut_words(path):
    texts = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            texts.append([word.word for word in pseg.cut(line.strip(), use_paddle=True)
                          if word.word not in stop_set and word.flag not in ['u', 'w', 'x', 'p', 'q', 'm', 'e', 'd']])
    save_data("../corpus/test_corpora_cut.txt", texts)


def merge_corpus(path1, path2, save_path):
    texts = [line for line in open(path1, 'r', encoding='utf-8').readlines()] \
           + [line for line in open(path2, 'r', encoding='utf-8').readlines()]
    with open(save_path, 'w', encoding='utf-8') as f:
        for text in texts:
            f.write(text)


def save_data(path, texts):
    with open(path, 'w', encoding='utf-8') as f:
        for text in texts:
            f.write(" ".join(text) + "\n")


# cut_words("../corpus/test_corpora.txt")
# cut_json("../corpus/all_data_data.json")
merge_corpus("../corpus/test_corpora_cut.txt", "../corpus/smp_cut.txt", "../corpus/all_cut.txt")
