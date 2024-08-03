from transformers import WhisperForConditionalGeneration, WhisperFeatureExtractor, WhisperTokenizerFast, WhisperProcessor
from peft import PeftModel, PeftConfig

save_directory = 'model'
lora_model = 'train/checkpoint-650'
peft_config = PeftConfig.from_pretrained(lora_model)
base_model = WhisperForConditionalGeneration.from_pretrained(peft_config.base_model_name_or_path)

model = PeftModel.from_pretrained(base_model, lora_model)
feature_extractor = WhisperFeatureExtractor.from_pretrained(peft_config.base_model_name_or_path)
tokenizer = WhisperTokenizerFast.from_pretrained(peft_config.base_model_name_or_path)
processor = WhisperProcessor.from_pretrained(peft_config.base_model_name_or_path)

model = model.merge_and_unload()
model.train(False)

model.save_pretrained(save_directory)
feature_extractor.save_pretrained(save_directory)
tokenizer.save_pretrained(save_directory)
processor.save_pretrained(save_directory)