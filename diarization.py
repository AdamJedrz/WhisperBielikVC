import sys
import os
from pyannote.audio import Pipeline
from pydub import AudioSegment

def main():
    if len(sys.argv) < 4 or len(sys.argv) > 5:
        print("Użycie:")
        print("python diarization.py <ścieżka_do_audio> <ścieżka_do_output_speakerów> <ścieżka_do_output_speechy> [liczba_mówców]")
        sys.exit(1)

    audio_path = sys.argv[1]
    output_speakers_dir = sys.argv[2]
    output_speech_dir = sys.argv[3]
    num_speakers = int(sys.argv[4]) if len(sys.argv) == 5 else None

    if not os.path.isfile(audio_path):
        print(f"Plik audio nie istnieje: {audio_path}")
        sys.exit(1)

    os.makedirs(output_speakers_dir, exist_ok=True)
    os.makedirs(output_speech_dir, exist_ok=True)

    hf_token = "xxxxxxxxxxxxxxxx"
    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", token=hf_token)

    audio = AudioSegment.from_wav(audio_path)

    ann = pipeline(audio_path, num_speakers=num_speakers)

    annotation = getattr(ann, "speaker_diarization", ann)

    segments = []
    for segment, _, speaker in annotation.itertracks(yield_label=True):
        segments.append({
            "speaker": str(speaker),
            "start": float(segment.start),
            "end": float(segment.end)
        })

    segments = sorted(segments, key=lambda x: x["start"])

    ordered = []
    for i, s in enumerate(segments, start=1):
        start_ms = int(s["start"] * 1000)
        end_ms   = int(s["end"] * 1000)

        snippet = audio[start_ms:end_ms]
        fname = os.path.join(output_speech_dir, f"speech_{i}_spk{s['speaker']}.wav")
        snippet.export(fname, format="wav")

        ordered.append((s["speaker"], snippet))

    speaker_audio = {}
    for spk, snip in ordered:
        if spk not in speaker_audio:
            speaker_audio[spk] = AudioSegment.silent(duration=0)
        speaker_audio[spk] += snip

    for spk, wav in speaker_audio.items():
        out_file = os.path.join(output_speakers_dir, f"speaker_{spk}.wav")
        wav.export(out_file, format="wav")

if __name__ == "__main__":
    main()
