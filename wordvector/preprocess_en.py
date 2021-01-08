import en_core_web_lg
import pandas as pd
from lxml import etree
import re


word_flag = ['ADJ', 'ADV']#, 'NOUN', 'VERB', 'INTJ']
nlp = en_core_web_lg.load()


def cut_xlsx(path):
    df = pd.read_excel(path, header=0)
    texts = df['内容'].values.reshape(-1).tolist()
    rates = df['星数'].values.reshape(-1).tolist()
    cut_text = []
    for doc in nlp.pipe(texts):
        s = []
        for token in doc:
            if not nlp.vocab[token.text].is_stop and token.pos_ in word_flag:
                s.append(token.text)
        cut_text.append(s)

    save_data_rates(path[:-5] + "_cut.txt", cut_text, rates)


def cut_xml(path):
    tree = etree.parse(path)
    root = tree.getroot()
    comments = root.xpath("//summary_trans")
    texts = [comment.text.strip() for comment in comments if comment.text is not None] + \
            [comment.text.strip() for comment in root.xpath("//text_trans") if comment.text is not None]

    cut_text = []
    for doc in nlp.pipe(texts):
        s = []
        for token in doc:
            if not nlp.vocab[token.text].is_stop and token.pos_ in word_flag:
                s.append(token.text)
        cut_text.append(s)

    save_data(path[:-4] + "_cut.txt", cut_text)


def cut_txt(path):
    texts = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            texts.append(line.strip())
    cut_text = []
    for doc in nlp.pipe(texts):
        s = []
        for token in doc:
            if not nlp.vocab[token.text].is_stop and token.pos_ in word_flag:
                s.append(token.text)
        cut_text.append(s)

    save_data(path[:-4] + "_cut.txt", cut_text)


def save_data(path, texts):
    with open(path, 'w', encoding='utf-8') as f:
        for text in texts:
            if not text:
                continue
            f.write(" ".join(text) + "\n")


def save_data_rates(path, texts, rates):
    with open(path, 'w', encoding='utf-8') as f:
        for i in range(len(texts)):
            if not texts[i]:
                continue
            f.write(" ".join(texts[i]) + "\t" + str(rates[i]) + "\n")


# cut_xlsx("../corpus/classics/classics_en.xlsx")
# cut_txt("../corpus/book/book_review.txt")
# cut_txt("../corpus/NYT_comment.txt")
# cut_xml("../corpus/classics/classics.xml")
import pandas as pd
df = pd.read_csv("D:/python/mylibs/Hotel_Reviews.csv")
pos = []
neg = []
for i in range(len(df)):
    n = df.loc[i, 'Negative_Review']
    p = df.loc[i, 'Positive_Review']
    if n != "No Negative":
        neg.append(n)
    if p != "No Positive":
        pos.append(p)

def cut_csv(arr, p, path):
    _cut_text = []
    for doc in nlp.pipe(arr):
        s = []
        for token in doc:
            if not nlp.vocab[token.text].is_stop and token.pos_ in word_flag:
                s.append(token.text)
        _cut_text.append(s)
    save(_cut_text, p, path)
def save(arr, p, path):
    with open(path, 'w', encoding='utf-8') as f:
        for i in arr:
            if " ".join(i) == "":
                continue
            f.write(" ".join(i)+"\t"+p+"\n")

# cut_csv(pos, 'p', "../corpus/hotel_en/pos.txt")
# cut_csv(neg, 'p', "../corpus/hotel_en/neg.txt")
cut_csv(neg[:8000], 'n', "../corpus/hotel_en/validation10000.tsv")
cut_csv(pos[:8000], 'p', "../corpus/hotel_en/pos_cut.tsv")
# save(pos+neg, 'n', "../corpus/hotel_en/all4train.txt")




