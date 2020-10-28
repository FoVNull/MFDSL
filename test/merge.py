# import os
# neg_text = set()
# for i in range(2999):
#     file = "neg."+str(i)+".txt"
#     with open("../corpus/hotel/neg/"+file, 'r', encoding='utf-8') as f:
#         s = "".join([line.strip() for line in f.readlines()])
#         neg_text.add(s.replace("\n", ""))
#
# pos_text = set()
# for i in range(2999):
#     file = "pos." + str(i) + ".txt"
#     with open("../corpus/hotel/pos/"+file, 'r', encoding='utf-8') as f:
#         s = "".join([line.strip() for line in f.readlines()])
#         pos_text.add(s.replace("\n", ""))
#
# with open("../corpus/hotel/neg.txt", 'w', encoding='utf-8') as f:
#     for i in neg_text:
#         if i == "":
#             continue
#         f.write(i+"\n")
#
# with open("../corpus/hotel/pos.txt", 'w', encoding='utf-8') as f:
#     for i in pos_text:
#         if i == "":
#             continue
#         f.write(i+"\n")
pos_text = [line.strip() for line in open("../corpus/hotel/pos.txt", 'r', encoding='utf-8').readlines()]
neg_text = [line.strip() for line in open("../corpus/hotel/neg.txt", 'r', encoding='utf-8').readlines()]

with open("../corpus/hotel/train.txt", 'w', encoding='utf-8') as f:
    for i in pos_text[:1300]:
        f.write(i+"\n")
    for i in neg_text[:1300]:
        f.write(i + "\n")

with open("../corpus/hotel/neg_test.txt", 'w', encoding='utf-8') as f:
    for i in pos_text[1300:]:
        f.write(i+"\n")

with open("../corpus/hotel/pos_test.txt", 'w', encoding='utf-8') as f:
    for i in neg_text[1300:]:
        f.write(i+"\n")