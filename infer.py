import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline


model_path = "model"
audio_file = "instrument_1.wav.reformatted.wav_10.wav_0000172480_0000363840.wav"
device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

processor = AutoProcessor.from_pretrained(model_path)
model = AutoModelForSpeechSeq2Seq.from_pretrained(model_path, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True)

prompt = torch.from_numpy(processor.get_prompt_ids("以下是普通话的句子。")).to(device)
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
)
generate_kwargs = {"task": "transcribe", "num_beams": 1, "language": "Chinese", "prompt_ids": prompt}
result = infer_pipe(audio_file, return_timestamps=True, generate_kwargs=generate_kwargs)


for chunk in result["chunks"]:
    print(f"[{chunk['timestamp'][0]}-{chunk['timestamp'][1]}s] {chunk['text']}")
