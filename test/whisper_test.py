import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

model_id = "openai/whisper-large-v3"

model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, low_cpu_mem_usage=True, use_safetensors=True
)

processor = AutoProcessor.from_pretrained(model_id)

# model.generation_config = model.generation_config.from_model_config(
#     model.config
# )
# model.generation_config.return_timestamps = False

model.to(device, dtype=torch_dtype)

pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    device=device,
    return_timestamps=True
)

result = pipe(
    "chair_sawtooth_chirp_1.mp3",
    generate_kwargs={
        "language": "pl",
        "task": "transcribe"
    },
)
print(result["text"])
