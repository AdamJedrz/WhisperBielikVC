import os
import re
from pydub import AudioSegment

from diarization import run_diarization
from generate_audio import generate_audio


TEMP_AUDIO = "temp_audio"
TEMP_SPEAKER = "temp_speaker"
TEMP_SPEECH = "temp_speech"
TEMP_TXT = "temp_txt_corrected"
TEMP_AUDIO_CORRECTED = "temp_audio_corrected"

INPUT_AUDIO = os.path.join(TEMP_AUDIO, "audio.wav")
RESULT_AUDIO = os.path.join(TEMP_AUDIO, "result.wav")


def extract_speaker_id(filename: str) -> int:
    match = re.search(r"speaker(\d+)", filename)
    if not match:
        raise ValueError(f"Nie można odczytać speakera z: {filename}")
    return int(match.group(1))


def main():

    #pyannote wyodrębnianie
    run_diarization(
        audio_path=INPUT_AUDIO,
        output_speakers_dir=TEMP_SPEAKER,
        output_speech_dir=TEMP_SPEECH
    )








    #generowanie wypowiedzi z obrobionego tekstu 
    os.makedirs(TEMP_AUDIO_CORRECTED, exist_ok=True)
    txt_files = sorted(os.listdir(TEMP_TXT))
    generated_wavs = []

    for txt in txt_files:
        speaker_id = extract_speaker_id(txt)

        speaker_wav = os.path.join(TEMP_SPEAKER, f"speaker{speaker_id}.wav")

        output_wav = os.path.join(TEMP_AUDIO_CORRECTED, txt.replace(".txt", ".wav"))

        generate_audio(
            text_path=os.path.join(TEMP_TXT, txt),
            speaker_wav=speaker_wav,
            output_wav=output_wav
        )

        generated_wavs.append(output_wav)

    #sklejenie wyniku
    final_audio = AudioSegment.silent(duration=0)
    for wav in generated_wavs:
        final_audio += AudioSegment.from_wav(wav)
    final_audio.export(RESULT_AUDIO, format="wav")

if __name__ == "__main__":
    main()
