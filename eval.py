import evaluate
import pandas as pd
from tqdm import tqdm
from decoder import LoraDecoder

cer = evaluate.load("cer")
decoder = LoraDecoder("medium", prompt="以下是普通话的句子。")

batch_size = 96
test_dataset = []
test_csv = "dataset/test.csv"

for index, item in pd.read_csv(test_csv, sep="\t").iterrows():
    test_dataset.append({"audio": item["audio"], "sentence": item["sentence"]})

for i in tqdm(range(0, len(test_dataset), batch_size)):
    audios = [item["audio"] for item in test_dataset[i : i + batch_size]]
    sentences = [item["sentence"] for item in test_dataset[i : i + batch_size]]
    results = decoder(audios)
    cer.add_batch(predictions=[item["text"] for item in results], references=sentences)
# 10.13
print(cer.compute() * 100)
