from peft import PeftModel, PeftConfig
from transformers import WhisperForConditionalGeneration, WhisperFeatureExtractor, WhisperTokenizer, WhisperProcessor

save_directory = "medium"
lora_model = "train/checkpoint-1100"
peft_config = PeftConfig.from_pretrained(lora_model)
base_model = WhisperForConditionalGeneration.from_pretrained(peft_config.base_model_name_or_path)

model = PeftModel.from_pretrained(base_model, lora_model)
feature_extractor = WhisperFeatureExtractor.from_pretrained(peft_config.base_model_name_or_path)
tokenizer = WhisperTokenizer.from_pretrained(peft_config.base_model_name_or_path)
processor = WhisperProcessor.from_pretrained(peft_config.base_model_name_or_path)

model = model.merge_and_unload()
model.train(False)

model.save_pretrained(save_directory)
feature_extractor.save_pretrained(save_directory)
tokenizer.save_pretrained(save_directory)
processor.save_pretrained(save_directory)
