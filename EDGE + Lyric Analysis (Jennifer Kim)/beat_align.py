import numpy as np
import librosa
import pickle
import glob
import os
from scipy.signal import argrelextrema

def compute_beat_alignment(motion_pkl_path, wav_path, fps=30):
    # --- music beats ---
    audio, sr = librosa.load(wav_path, sr=None)
    tempo, beat_frames = librosa.beat.beat_track(y=audio, sr=sr)
    beat_times = librosa.frames_to_time(beat_frames, sr=sr)

    # --- motion beats ---
    with open(motion_pkl_path, "rb") as f:
        data = pickle.load(f)

    full_pose = data["full_pose"]  # (T, 24, 3)
    vel = np.linalg.norm(full_pose[1:] - full_pose[:-1], axis=-1)  # (T-1, 24)
    vel_total = vel.sum(axis=-1)  # (T-1,)

    motion_beat_frames = argrelextrema(vel_total, np.greater, order=5)[0]
    motion_beat_times = motion_beat_frames / fps

    if len(beat_times) == 0 or len(motion_beat_times) == 0:
        return 0.0

    distances = [np.min(np.abs(beat_times - mt)) for mt in motion_beat_times]
    beat_align = np.mean(np.array(distances) < 0.1)
    return beat_align

if __name__ == "__main__":
    motion_dir = "./motions"
    music_dir = "./music"

    for pkl in glob.glob(f"{motion_dir}/*.pkl"):
        # match pkl name to wav name
        # pkl is like test_clap_youre_hands_clipped.pkl
        # wav is like clap_youre_hands_clipped.wav
        basename = os.path.basename(pkl).replace(".pkl", "")
        song_name = "_".join(basename.split("_")[1:])  # strip "test_" prefix
        wav_path = os.path.join(music_dir, song_name + ".wav")

        if not os.path.exists(wav_path):
            print(f"WAV not found for {basename}, skipping")
            continue

        score = compute_beat_alignment(pkl, wav_path)
        print(f"{song_name}: Beat Alignment = {score:.4f}")

