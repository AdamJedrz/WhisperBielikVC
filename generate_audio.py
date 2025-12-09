import sys
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

def print_usage():
    print("\nBŁĄD: Nie podano wymaganych argumentów!\n")
    print("Użycie:")
    print("  python generate_audio.py <plik_audio.wav> <plik_wyjściowy.wav> <plik_tekstu.txt>\n")
    print("Przykład:")
    print("  python generate_audio.py test/Lektor.wav output.wav text.txt\n")


def main():
    if len(sys.argv) != 4:
        print_usage()
        sys.exit(1)

    speaker_wav = sys.argv[1]
    output_path = sys.argv[2]
    text_file = sys.argv[3]

    try:
        with open(text_file, "r", encoding="utf-8") as f:
            text = f.read().strip()
    except FileNotFoundError:
        print(f"\nBŁĄD: Nie znaleziono pliku tekstowego: {text_file}\n")
        sys.exit(1)

    with torch.serialization.safe_globals(safe_classes):
        tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to("cuda")

    tts.tts_to_file(
        text=text,
        speaker_wav=speaker_wav,
        file_path=output_path,
        language="pl"
    )

    print(f"\nZapisano: {output_path}\n")

if __name__ == "__main__":
    main()
