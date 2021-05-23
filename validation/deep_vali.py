import argparse
from sklearn.model_selection import train_test_split
import pickle
from kashgari.tasks.classification import BiLSTM_Model, CNN_Model
from kashgari.embeddings import WordEmbedding


def data_load(path):
    x = []
    y = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            sen, label = line.strip().split("\t")
            y.append(label)
            x.append(sen.split(" "))
    return x, y


def build_vec(path, dim):
    if len(path) > 1:
        pkl_vec = pickle.load(open(path[0], 'rb'))
        pkl_vec2 = pickle.load(open(path[1], 'rb'))
    else:
        pkl_vec = pickle.load(open(path[0], 'rb'))
    vectors = []
    for item in pkl_vec.items():
        if len(path) > 1:
            if item[0] not in pkl_vec2.keys():
                continue
                # v = [str(_) for _ in item[1]] + ['0.0' for _ in range(100)]
            else:
                v = [str(_) for _ in item[1]] + [str(_) for _ in pkl_vec2[item[0]]]
        else:
            v = [str(_) for _ in item[1]]
        vectors.append(item[0] + " " + " ".join(v) + "\n")

    with open("temp.vec", 'w', encoding='utf-8') as _f:
        _f.write("{} {}\n".format(len(vectors), dim))
        for v in vectors:
            _f.write(v)


if __name__ == '__main__':
    parse = argparse.ArgumentParser(description="validate sentiment dictionary in deep learning")
    parse.add_argument("--model", default="BiLSTM",
                       help="you can choose [BiLSTM, CNN]")
    parse.add_argument("--corpus", type=str, default="../corpus/amazon/book/vali2000.tsv", help="specify corpus")
    parse.add_argument("--dimension", default=200, type=int,
                       help="dimension of dictionary")
    args = parse.parse_args()

    metrics = {"acc": 0.0, "f1": 0.0, "max_acc": 0.0, "max_f1": 0.0}

    x_data, y_data = data_load(args.corpus)
    for seed in range(1):
        x_train, x_test, y_train, y_test = train_test_split(
            x_data, y_data,
            test_size=0.5, random_state=seed, shuffle=True
        )

        build_vec(['../reference/output/wv.pkl', '../reference/output/sv.pkl'], args.dimension)
        embedding = WordEmbedding("temp.vec")
        model = globals()[args.model + "_Model"](embedding)
        model.fit(x_train, y_train, batch_size=64, epochs=30, callbacks=None, fit_kwargs=None)

        report = model.evaluate(x_test, y_test, batch_size=64, digits=4, truncating=True)
        acc = float(report['detail']['accuracy'])
        f1 = float(report['detail']['macro avg']['f1-score'])
        metrics["acc"] += acc
        metrics["f1"] += f1
        metrics["max_acc"] = max(metrics["max_acc"], acc)
        metrics["max_f1"] = max(metrics["max_f1"], f1)
        print(seed+1, metrics)
    with open("deep_res.txt", 'a+', encoding="utf-8") as f:
        f.write(
            "{}\t{}\t{}\t{}\n".format(metrics['acc'] / 10, metrics['f1'] / 10,
                                      metrics['max_acc'], metrics['max_f1'])
        )
    # model.save("./deep_output")
