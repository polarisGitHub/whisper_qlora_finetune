import csv
import codecs


def common_voice_dataset(input_file: str, csv_file: str, audio_prefix: str) -> None:
    with codecs.open(csv_file, mode="w", encoding="utf-8") as wf:
        writer = csv.writer(wf, delimiter="\t", quotechar='"')
        writer.writerow(["audio", "sentence"])
        with codecs.open(input_file, mode="r", encoding="utf-8") as rf:
            reader = csv.DictReader(rf, delimiter="\t")
            for row in reader:
                up_votes = int(row["up_votes"])
                down_votes = int(row["down_votes"])
                if up_votes > 2 and down_votes < 1:
                    writer.writerow(
                        [
                            audio_prefix +  row["path"],
                            row["sentence"],
                        ]
                    )


if __name__ == "__main__":
    prefix = "/home/polaris_he/cv-corpus-12.0-2022-12-07/zh-CN/clips/"
    common_voice_dataset("csv/validated.tsv", "dataset/validated.csv", prefix)
    common_voice_dataset("csv/train.tsv", "dataset/train.csv", prefix)
    common_voice_dataset("csv/test.tsv", "dataset/test.csv", prefix)
