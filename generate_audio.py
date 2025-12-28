import torch
from TTS.api import TTS

import TTS.tts.configs.xtts_config as xtts_config
import TTS.tts.models.xtts as xtts_model
import TTS.config.shared_configs as shared_configs

safe_classes = [
    xtts_config.XttsConfig,
    xtts_model.XttsAudioConfig,
    xtts_model.XttsArgs,
    shared_configs.BaseDatasetConfig
]

_tts_model = None


def get_tts():
    global _tts_model
    if _tts_model is None:
        with torch.serialization.safe_globals(safe_classes):
            _tts_model = TTS(
                "tts_models/multilingual/multi-dataset/xtts_v2"
            ).to("cuda")
    return _tts_model


def generate_audio(
    text_path: str,
    speaker_wav: str,
    output_wav: str,
    language: str = "pl"
):
    with open(text_path, "r", encoding="utf-8") as f:
        text = f.read().strip()

    tts = get_tts()

    tts.tts_to_file(
        text=text,
        speaker_wav=speaker_wav,
        file_path=output_wav,
        language=language
    )
