import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline


class LoraDecoder(object):

    def __init__(self, lora_model: str, prompt: str = None) -> None:
        self.lora_model = lora_model
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        self.processor = AutoProcessor.from_pretrained(self.lora_model)
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            self.lora_model,
            torch_dtype=self.torch_dtype,
            low_cpu_mem_usage=True,
            use_safetensors=True,
        )

        self.generate_kwargs = {"task": "transcribe", "num_beams": 1, "language": "Chinese", "prompt_ids": prompt}
        if prompt is not None:
            self.generate_kwargs["prompt_ids"] = torch.from_numpy(self.processor.get_prompt_ids(prompt)).to(self.device)

        self.infer_pipe = pipeline(
            "automatic-speech-recognition",
            model=self.model,
            tokenizer=self.processor.tokenizer,
            feature_extractor=self.processor.feature_extractor,
            max_new_tokens=128,
            chunk_length_s=30,
            batch_size=8,
            torch_dtype=self.torch_dtype,
            device=self.device,
        )

    def __call__(self, audio_files: list[str], return_timestamps: bool = False) -> list[dict]:
        return self.infer_pipe(audio_files, return_timestamps=return_timestamps, generate_kwargs=self.generate_kwargs)
