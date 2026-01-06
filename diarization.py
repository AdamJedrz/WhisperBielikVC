import os

import torch

from pyannote.audio import Pipeline
from pydub import AudioSegment


def run_diarization(
    audio_path: str,
    output_speakers_dir: str,
    output_speech_dir: str
):
    os.makedirs(output_speakers_dir, exist_ok=True)
    os.makedirs(output_speech_dir, exist_ok=True)

    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1")

    pipeline.segmentation.threshold = 0.5
    pipeline.clustering.threshold = 0.5
    pipeline.segmentation.min_duration_on = 0
    pipeline.segmentation.min_duration_off = 0

    audio = AudioSegment.from_wav(audio_path)
    ann = pipeline(audio_path, num_speakers=2) #dobrac do potrzeb
    annotation = getattr(ann, "speaker_diarization", ann)

    segments = []
    for segment, _, speaker in annotation.itertracks(yield_label=True):
        segments.append({
            "speaker": str(speaker),
            "start": float(segment.start),
            "end": float(segment.end)
        })

    segments.sort(key=lambda x: x["start"])

    speaker_map = {}
    speaker_counter = 1

    ordered_segments = []

    for i, s in enumerate(segments, start=1):
        if s["speaker"] not in speaker_map:
            speaker_map[s["speaker"]] = speaker_counter
            speaker_counter += 1

        spk_id = speaker_map[s["speaker"]]

        start_ms = int(s["start"] * 1000)
        end_ms = int(s["end"] * 1000)

        snippet = audio[start_ms:end_ms]

        fname = f"speech{i}_speaker{spk_id}.wav"
        snippet.export(
            os.path.join(output_speech_dir, fname),
            format="wav"
        )

        ordered_segments.append((spk_id, snippet))

    speaker_segments = {}

    for spk_id, snip in ordered_segments:
        speaker_segments.setdefault(spk_id, [])
        speaker_segments[spk_id].append(snip)

    for spk_id, snippets in speaker_segments.items():
        best_snippet = max(snippets, key=lambda x: len(x))

        out_file = os.path.join(
            output_speakers_dir,
            f"speaker{spk_id}.wav"
        )
        best_snippet.export(out_file, format="wav")
