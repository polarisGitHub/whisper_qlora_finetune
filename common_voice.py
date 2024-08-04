import csv
import codecs
import random


input_tsv = "csv/validated.tsv"
train_csv = "dataset/train.csv"
test_csv = "dataset/test.csv"
prefix = "/home/polaris_he/cv-corpus-12.0-2022-12-07/zh-CN/clips/"

results = []
with codecs.open(input_tsv, mode="r", encoding="utf-8") as rf:
    reader = csv.DictReader(rf, delimiter="\t")
    for row in reader:
        up_votes = int(row["up_votes"])
        down_votes = int(row["down_votes"])
        if up_votes > 2 and down_votes < 1:
            results.append(
                [
                    prefix + row["path"],
                    row["sentence"],
                ]
            )

N = 10
split_index = int((N - 1) * len(results) / N)
random.shuffle(results)

train = results[:split_index]
test = results[split_index:]


def write_dataset(output: str, dataset: list[dict]) -> None:
    with codecs.open(output, mode="w", encoding="utf-8") as wf:
        writer = csv.writer(wf, delimiter="\t", quotechar='"')
        writer.writerow(["audio", "sentence"])
        writer.writerows(dataset)


write_dataset("dataset/train.csv", train)
write_dataset("dataset/test.csv", test)
