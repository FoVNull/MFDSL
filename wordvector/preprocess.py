import json
import jieba.posseg as pseg
from lxml import etree

stop_set = set(line.strip() for line in open("../reference/HIT_stopwords.txt", 'r', encoding='utf-8').readlines())
word_flag = ['Ag', 'a', 'ad', 'an', 'Dg', 'd', 'i', 'l', 'z']


def cut_json(path):
    texts = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            dic = json.loads(line.strip())
            texts.append([word.word for word in pseg.cut(dic['content'], use_paddle=True)
                          if word.word not in stop_set and word.flag in word_flag])
    save_data(path[:-5] + "_cut.txt", texts)


def cut_words(path, p):
    texts = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            texts.append([word.word for word in pseg.cut(line.strip(), use_paddle=True)
                          if word.word not in stop_set and word.flag in word_flag])

    # save_data_rates(path[:-4] + "_cut.txt", texts,  [p]*len(texts))
    save_data(path[:-4] + "_cut.txt", texts)


def cut_csv(path):
    texts = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            text = line.strip().split(",")[1]
            texts.append([word.word for word in pseg.cut(text, use_paddle=True)
                          if word.word not in stop_set and word.flag in word_flag])
    save_data(path[:-4] + "_cut.txt", texts)


def cut_tsv(path):
    texts = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            text = line.strip().split("\t")[0]
            texts.append(([word.word for word in pseg.cut(text, use_paddle=True)
                           if word.word not in stop_set and word.flag in word_flag],
                          line.strip().split("\t")[1]))
    with open(path[:-4] + "_cut.tsv", 'w', encoding='utf-8') as f:
        for text in texts:
            if text[0] == "":
                continue
            f.write(" ".join(text[0]) + "\t" + text[1] + "\n")
    save_data(path[:-4] + "_cut.txt", [tp[0] for tp in texts])


def merge_corpus(path1, path2, save_path):
    texts = [line for line in open(path1, 'r', encoding='utf-8').readlines()] \
            + [line for line in open(path2, 'r', encoding='utf-8').readlines()]
    with open(save_path, 'w', encoding='utf-8') as f:
        for text in texts:
            f.write(text)


def cut_xml(path, tag):
    tree = etree.parse(path, parser=etree.HTMLParser(encoding='utf-8'))
    root = tree.getroot()
    comments = root.xpath(tag)
    # rates = root.xpath("//score")
    texts = []
    for comment in comments:
        if comment.text is None:
            continue
        texts.append([word.word for word in pseg.cut(comment.text.strip(), use_paddle=True)
                      if word.word not in stop_set])
    # save_data_rates(path[:-4] + "_zh_cut.txt", texts, [rate.text for rate in rates])
    save_data(path[:-4] + ".txt", texts)


def save_data(path, texts):
    with open(path, 'w', encoding='utf-8') as f:
        for text in texts:
            f.write(" ".join(text) + "\n")


def save_data_rates(path, texts, rates):
    with open(path, 'w', encoding='utf-8') as f:
        for i in range(len(texts)):
            if not texts[i]:
                continue
            f.write(" ".join(texts[i]) + "\t" + str(rates[i]) + "\n")


# cut_xml("../corpus/classics/classics.xml", "//text")
# cut_tsv("../corpus/hotel/train.tsv")
# cut_words("../corpus/hotel/pos.txt", 'p')
# cut_words("../corpus/hotel/neg.txt", 'n')
# cut_json("../corpus/all_data_data.json")
# cut_words("../corpus/hotel/all.txt", 'p')
# merge_corpus("../corpus/hotel/neg_cut.txt", "../corpus/hotel/pos_cut.txt", "../corpus/hotel/all_cut.tsv")
# merge_corpus("../corpus/hotel/neg.txt", "../corpus/hotel/pos.txt", "../corpus/hotel/all.txt")
# cut_xml("../corpus/NLPIR_weibo.xml", "//article")
# merge_corpus("../corpus/hotel/all4train.txt", "../corpus/NLPIR_weibo.txt", "../corpus/zh_train.txt")
cut_words("../corpus/hotel/all4000.txt", 'p')
text = [line.strip() for line in open("../corpus/hotel/all4000_cut.txt")]
with open("../corpus/hotel/test4000.tsv", 'w', encoding='utf-8') as f:
    for i in text[:2000]:
        f.write(i+"\tn\n")
    for i in text[2000:]:
        f.write(i+"\tp\n")

