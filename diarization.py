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

    hf_token = "xxxxxxxxxxxxxxxxxxxxx" #wpisz token

    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", token=hf_token)
    audio = AudioSegment.from_wav(audio_path)

    ann = pipeline(audio_path, num_speakers=num_speakers)
    annotation = getattr(ann, "speaker_diarization", ann)

    segments = []
    for segment, _, label in annotation.itertracks(yield_label=True):
        segments.append({"speaker": str(label), "start": float(segment.start), "end": float(segment.end)})
    segments = sorted(segments, key=lambda x: x["start"])

    window = 0.25
    new_segments = []
    for s in segments:
        dur = s["end"] - s["start"]
        if dur > window:
            cur = s["start"]
            while cur < s["end"]:
                end = min(s["end"], cur + window)
                new_segments.append({"speaker": s["speaker"], "start": cur, "end": end})
                cur = end
        else:
            new_segments.append(s)
    segments = new_segments

    merged_segments = []
    current = None
    for seg in sorted(segments, key=lambda x: x["start"]):
        if current is None:
            current = seg
        elif seg["speaker"] == current["speaker"] and seg["start"] - current["end"] < 0.1:
            current["end"] = seg["end"]
        else:
            merged_segments.append(current)
            current = seg
    if current:
        merged_segments.append(current)
    segments = merged_segments

    ordered = []
    for i, s in enumerate(segments, start=1):
        start_ms = int(s["start"] * 1000)
        end_ms   = int(s["end"] * 1000)
        snippet = audio[start_ms:end_ms]
        fname = os.path.join(output_speech_dir, f"speech{i}.wav")
        snippet.export(fname, format="wav")
        ordered.append((s["speaker"], snippet))

    speaker_audio = {}
    for spk, snip in ordered:
        speaker_audio.setdefault(spk, AudioSegment.silent(duration=0))
        speaker_audio[spk] += snip

    for i, (spk, wav) in enumerate(speaker_audio.items(), start=1):
        wav.export(os.path.join(output_speakers_dir, f"speaker{i}.wav"), format="wav")


if __name__ == "__main__":
    main()
