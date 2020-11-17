import en_core_web_lg
import pandas as pd
from lxml import etree


word_flag = ['ADJ', 'ADV', 'NOUN', 'PROPN', 'VERB', 'X']
nlp = en_core_web_lg.load()


def cut_xlsx(path):
    df = pd.read_excel(path, header=0)
    texts = [i for i in df['内容'].values.reshape(-1)]
    cut_text = []
    for doc in nlp.pipe(texts):
        s = []
        for token in doc:
            if not nlp.vocab[token.text].is_stop and token.pos_ in word_flag:
                s.append(token.text)
        cut_text.append(s)

    save_data(path[:-5] + "_cut.txt", cut_text)


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


def save_data(path, texts):
    with open(path, 'w', encoding='utf-8') as f:
        for text in texts:
            if not text:
                continue
            f.write(" ".join(text) + "\n")


# cut_xlsx("../corpus/classics/classics_en.xlsx")
cut_xml("../corpus/classics/classics.xml")