import whisper
import numpy as np
import json
from sentence_transformers import SentenceTransformer

def get_lyric_emb_for_slice(segments, slice_idx, slice_duration=2.5, embed_dim=384):
    window_start = slice_idx * slice_duration
    window_end = window_start + slice_duration
    relevant = [
        np.array(s["embedding"]) for s in segments
        if s["start"] < window_end and s["end"] > window_start
    ]
    if relevant:
        return np.mean(relevant, axis=0).astype(np.float32)
    return np.zeros(embed_dim, dtype=np.float32)

# --- config ---
wav_file = "./music/dance_with_instruction.wav"  # change to any of your songs
out_length = 15  # seconds
slice_duration = 2.5
num_slices = int(out_length / slice_duration)

# --- Step 1: Whisper ---
print("Transcribing...")
whisper_model = whisper.load_model("base")
result = whisper_model.transcribe(wav_file, word_timestamps=True)

# --- Step 2: SentenceTransformer ---
print("Encoding...")
encoder = SentenceTransformer("all-MiniLM-L6-v2")
segments = []
for seg in result["segments"]:
    emb = encoder.encode(seg["text"].strip())
    segments.append({
        "text": seg["text"].strip(),
        "start": round(seg["start"], 2),
        "end": round(seg["end"], 2),
        "embedding_preview": emb[:5].tolist()  # first 5 numbers only for readability
    })

# --- Step 3: Slice matching ---
slice_report = []
for i in range(num_slices):
    window_start = i * slice_duration
    window_end = window_start + slice_duration
    matched = [s["text"] for s in segments
               if s["start"] < window_end and s["end"] > window_start]
    emb = get_lyric_emb_for_slice(
        [{**s, "embedding": np.array(s["embedding_preview"])} for s in segments], i
    )
    slice_report.append({
        "slice": i,
        "window": f"{window_start:.1f}s - {window_end:.1f}s",
        "matched_lyrics": matched if matched else ["(no lyrics)"],
        "embedding_preview": emb[:5].tolist()
    })

# --- Save results ---
output = {
    "song": wav_file,
    "whisper_segments": segments,
    "slice_assignments": slice_report
}

with open("lyric_inspection.json", "w") as f:
    json.dump(output, f, indent=2)

print("Saved to lyric_inspection.json")