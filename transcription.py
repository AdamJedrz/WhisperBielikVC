import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

def load_stt_model():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    # torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    model_id = "openai/whisper-large-v3"

    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id, low_cpu_mem_usage=True, use_safetensors=True
    )
    model.to(device)

    processor = AutoProcessor.from_pretrained(model_id)

    return model, processor


def transcribe_audio(model, processor, language, input_path) -> str:
    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        # dtype=model.dtype,
        device=model.device, 
        # language=language,
        return_timestamps=True
    )

    return pipe(input_path)['text']