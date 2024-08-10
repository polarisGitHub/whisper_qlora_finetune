import evaluate
import pandas as pd
from tqdm import tqdm
from decoder import Decoder

# TODO use dataset
cer = evaluate.load("cer")
decoder = Decoder("large-v3", prompt="以下是普通话的句子。")
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

# 19.91 openai/whisper-medium
# 16.79 openai/whisper-large-v3
# 10.13 medium-lora
# 8.44 large-v3-lora
print(cer.compute() * 100)
