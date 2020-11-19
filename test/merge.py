import random
texts = []
with open("../corpus/classics/classics_en_cut.txt", 'r', encoding='utf-8') as f:
    for line in f.readlines():
        texts.append(line.strip())

random.shuffle(texts)

with open("../corpus/classics/classics_en_all.txt", 'w', encoding='utf-8') as f:
    for i in texts:
        f.write(i+"\n")
# with open("../corpus/classics/classics_en_va.txt", 'w', encoding='utf-8') as f:
#     for i in texts[10000:]:
#         f.write(i+"\n")