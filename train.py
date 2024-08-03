import torch
import evaluate
import pandas as pd
from datasets import Audio, Dataset
from dataclasses import dataclass
from typing import Any, Dict, List, Union
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    WhisperFeatureExtractor,
    WhisperTokenizer,
    BitsAndBytesConfig,
)


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch


if __name__ == "__main__":
    batch_size = 48
    train_sample = 15000 # 电脑跑不动全量
    train_csv = "dataset/validated.csv"
    test_csv = "dataset/test.csv"

    language = "zh"
    task = "transcribe"
    sampling_rate = 16000
    model_name_or_path = "openai/whisper-large-v3"

    model = WhisperForConditionalGeneration.from_pretrained(
        model_name_or_path,
        quantization_config=BitsAndBytesConfig(load_in_8bit=True),
        device_map="auto",
    )
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []
    model = prepare_model_for_kbit_training(model)

    feature_extractor = WhisperFeatureExtractor.from_pretrained(model_name_or_path)
    tokenizer = WhisperTokenizer.from_pretrained(model_name_or_path, language=language, task=task)
    processor = WhisperProcessor.from_pretrained(model_name_or_path, language=language, task=task)
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

    train_pd = pd.read_csv(train_csv, sep="\t").sample(train_sample)
    test_pd = pd.read_csv(test_csv, sep="\t")
    train_dataset = Dataset.from_pandas(train_pd).cast_column("audio", Audio(sampling_rate=sampling_rate))
    test_dataset = Dataset.from_pandas(test_pd).cast_column("audio", Audio(sampling_rate=sampling_rate))

    def prepare_dataset(examples):
        audio = examples["audio"]
        examples["input_features"] = feature_extractor(audio["array"], sampling_rate=sampling_rate).input_features[0]
        sentences = examples["sentence"]
        examples["labels"] = tokenizer(sentences).input_ids
        del examples["sentence"], examples["audio"]
        return examples

    train_dataset = train_dataset.map(prepare_dataset, num_proc=1)
    test_dataset = test_dataset.map(prepare_dataset, num_proc=1)

    config = LoraConfig(
        r=32,
        lora_alpha=64,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
    )
    model = get_peft_model(model, config)
    model.print_trainable_parameters()

    # eval爆显存，内存
    # metric = evaluate.load("wer")
    # def compute_metrics(pred):
    #     pred_ids = pred.predictions
    #     label_ids = pred.label_ids

    #     # replace -100 with the pad_token_id
    #     label_ids[label_ids == -100] = tokenizer.pad_token_id

    #     # we do not want to group tokens when computing the metrics
    #     pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    #     label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    #     wer = 100 * metric.compute(predictions=pred_str, references=label_str)

    #     return {"wer": wer}

    training_args = Seq2SeqTrainingArguments(
        output_dir="train/",
        num_train_epochs=10,
        per_device_train_batch_size=batch_size,
        # increase by 2x for every 2x decrease in batch size
        gradient_accumulation_steps=1,
        learning_rate=1e-3,
        warmup_steps=100,
        eval_strategy="steps",
        # gradient_checkpointing=True,
        # optim="adamw_torch",
        fp16=True,
        dataloader_num_workers=4,
        per_device_eval_batch_size=16,
        generation_max_length=255,
        save_steps=50,
        eval_steps=50,
        logging_steps=10,
        report_to=["tensorboard"],
        load_best_model_at_end=True,
        # eval_accumulation_steps=10,
        # required as the PeftModel forward doesn't have the signature of the wrapped model's forward
        remove_unused_columns=False,
        label_names=["labels"],  # same reason as above
        push_to_hub=False,
    )

    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        data_collator=data_collator,
        # compute_metrics=compute_metrics,
        tokenizer=processor.feature_extractor,
    )

    processor.save_pretrained(training_args.output_dir)
    model.config.use_cache = False
    trainer.train()
