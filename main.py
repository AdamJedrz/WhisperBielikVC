import os
import re
import torch
from pydub import AudioSegment

from diarization import run_diarization
from generate_audio import generate_audio
from transcription import load_stt_model, transcribe_audio
from commit_big_think import load_llm, big_think

PROMPT = "prompt_dialog.txt"

TEMP_AUDIO = "temp_audio"
TEMP_SPEAKER = "temp_speaker"
TEMP_SPEECH = "temp_speech"
TEMP_TXT = "temp_txt_corrected"
TEMP_RAW = "temp_txt_raw"
TEMP_AUDIO_CORRECTED = "temp_audio_corrected"

# INPUT_AUDIO = os.path.join(TEMP_AUDIO, "audio.wav")
# RESULT_AUDIO = os.path.join(TEMP_AUDIO, "result.wav")
INPUT_AUDIO = os.path.join(TEMP_AUDIO, "dialog4.wav")
RESULT_AUDIO = os.path.join(TEMP_AUDIO, "out.wav")

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

    #whisper STT
    stt_model, processor = load_stt_model()
    for audio_filename in os.listdir(TEMP_SPEECH):
        transcript_text = transcribe_audio(stt_model, processor, "pl", os.path.join(TEMP_SPEECH, audio_filename))
        transcript_filename = os.path.join(TEMP_RAW, audio_filename.split('.')[0]+".txt")

        mode = 'x'
        if os.path.exists(transcript_filename):
            mode = 'w'

        with open(transcript_filename, mode, encoding="utf-8") as transcript_file:
            transcript_file.write(transcript_text)

    del stt_model
    torch.cuda.empty_cache()

    #Bielik
    llm, tokenizer = load_llm()
    dialog_lines = []

    prev_speaker = None

    for raw_text_filename in sorted(os.listdir(TEMP_RAW)):
        raw_path = os.path.join(TEMP_RAW, raw_text_filename)
        if not os.path.isfile(raw_path):
            continue

        parts = raw_text_filename.split("_speaker")
        if len(parts) != 2:
            continue

        speaker = parts[1].replace(".txt", "")
        
        with open(raw_path, "r", encoding="utf-8") as f:
            text = f.read().strip()
        if not text:
            continue

        if speaker == prev_speaker:
            dialog_lines[-1] += " " + text
        else:
            dialog_lines.append(f"- {text}")
            prev_speaker = speaker

    dialog_text = "\n".join(dialog_lines)

    refined_dialog = big_think(llm, tokenizer, PROMPT, dialog_text, 0.7)
    output_path = os.path.join(TEMP_RAW, "dialog_refined.txt")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(refined_dialog)




    # Rozbijanie dialog_refined.txt na wypowiedzi

    refined_dialog_path = os.path.join(TEMP_RAW, "dialog_refined.txt")
    with open(refined_dialog_path, "r", encoding="utf-8") as f:
        refined_text = f.read()

    utterances = [
        line[2:].strip()
        for line in refined_text.splitlines()
        if line.strip().startswith("- ")
    ]

    speech_files = sorted(
        (f for f in os.listdir(TEMP_SPEECH) if f.endswith(".wav")),
        key=lambda x: int(re.search(r'speech(\d+)_', x).group(1))
    )

    if not speech_files:
        raise ValueError("Brak plików audio w TEMP_SPEECH")

    first_speaker_id = extract_speaker_id(speech_files[0])
    second_speaker_id = 1 if first_speaker_id != 1 else 2

    speakers = [first_speaker_id, second_speaker_id]
    speaker_index = 0

    for i, (utterance, speech_file) in enumerate(zip(utterances, speech_files)):
        speaker_id = speakers[speaker_index]
        speaker_index = 1 - speaker_index

        if i == 0:
            base_name = os.path.splitext(speech_file)[0]
            output_txt_path = os.path.join(TEMP_TXT, f"{base_name}.txt")
        else:
            output_txt_path = os.path.join(TEMP_TXT, f"speech{i+1}_speaker{speaker_id}.txt")

        with open(output_txt_path, "w", encoding="utf-8") as f:
            f.write(utterance)
    




    #generowanie wypowiedzi z obrobionego tekstu 
    os.makedirs(TEMP_AUDIO_CORRECTED, exist_ok=True)
    txt_files = sorted(
        os.listdir(TEMP_TXT),
        key=lambda x: int(re.search(r'speech(\d+)_', x).group(1))
    )
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
