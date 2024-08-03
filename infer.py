import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

model_path = "model"
audio_file = "/home/polaris_he/cv-corpus-12.0-2022-12-07/zh-CN/clips/common_voice_zh-CN_32947961.mp3"
device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

processor = AutoProcessor.from_pretrained(model_path)
model = AutoModelForSpeechSeq2Seq.from_pretrained(model_path, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True)

infer_pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    max_new_tokens=128,
    chunk_length_s=30,
    batch_size=8,
    torch_dtype=torch_dtype,
    device=device,
    model_kwargs={"initial_prompt":"以下是普通话的句子。"},
)
generate_kwargs = {"task": "transcribe", "num_beams": 1, "language": "Chinese"}
result = infer_pipe(audio_file, return_timestamps=True, generate_kwargs=generate_kwargs)


for chunk in result["chunks"]:
    print(f"[{chunk['timestamp'][0]}-{chunk['timestamp'][1]}s] {chunk['text']}")
