import glob
import os
from functools import cmp_to_key
from pathlib import Path
from tempfile import TemporaryDirectory
import random

import jukemirlib
import numpy as np
import torch
from tqdm import tqdm

from args import parse_test_opt
from data.slice import slice_audio
from EDGE import EDGE
from data.audio_extraction.baseline_features import extract as baseline_extract
from data.audio_extraction.jukebox_features import extract as juke_extract


#-------------JK Edits-------------
import torch.distributed as dist
import whisper
from sentence_transformers import SentenceTransformer
from lyric_fuser import LyricFuser
#----------------------------------


# sort filenames that look like songname_slice{number}.ext
key_func = lambda x: int(os.path.splitext(x)[0].split("_")[-1].split("slice")[-1])


def stringintcmp_(a, b):
    aa, bb = "".join(a.split("_")[:-1]), "".join(b.split("_")[:-1])
    ka, kb = key_func(a), key_func(b)
    if aa < bb:
        return -1
    if aa > bb:
        return 1
    if ka < kb:
        return -1
    if ka > kb:
        return 1
    return 0


stringintkey = cmp_to_key(stringintcmp_)


#-------------JK Edits-------------
def get_lyric_emb_for_slice(segments, slice_idx, slice_duration=2.5, embed_dim=384):
    """Get averaged lyric embedding for a given slice index."""
    window_start = slice_idx * slice_duration
    window_end = window_start + slice_duration
    relevant = [
        np.array(s["embedding"]) for s in segments
        if s["start"] < window_end and s["end"] > window_start
    ]
    if relevant:
        return np.mean(relevant, axis=0).astype(np.float32)
    return np.zeros(embed_dim, dtype=np.float32)
#----------------------------------



def test(opt):
    feature_func = juke_extract if opt.feature_type == "jukebox" else baseline_extract
    sample_length = opt.out_length
    sample_size = int(sample_length / 2.5) - 1

    temp_dir_list = []
    all_cond = []
    all_filenames = []

    if not dist.is_initialized():
        dist.init_process_group(backend='gloo', init_method='tcp://localhost:23456', rank=0, world_size=1)
    
    
    if opt.use_cached_features:
        print("Using precomputed features")
        # all subdirectories
        dir_list = glob.glob(os.path.join(opt.feature_cache_dir, "*/"))
        for dir in dir_list:
            file_list = sorted(glob.glob(f"{dir}/*.wav"), key=stringintkey)
            juke_file_list = sorted(glob.glob(f"{dir}/*.npy"), key=stringintkey)
            assert len(file_list) == len(juke_file_list)
            # random chunk after sanity check
            rand_idx = random.randint(0, len(file_list) - sample_size)
            file_list = file_list[rand_idx : rand_idx + sample_size]
            juke_file_list = juke_file_list[rand_idx : rand_idx + sample_size]
            cond_list = [np.load(x) for x in juke_file_list]
            all_filenames.append(file_list)
            all_cond.append(torch.from_numpy(np.array(cond_list)))
    else:
        print("Computing features for input music")

        #-------------JK Edits-------------
        whisper_model = whisper.load_model("base")
        lyric_encoder = SentenceTransformer("all-MiniLM-L6-v2")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
       
        fuser = LyricFuser(lyric_dim=384, music_dim=4800).to(device)  #music slice dimension = 4800
        fuser.gate.data.fill_(10.0)  ## Weight for word embedding
        #----------------------------------


        for wav_file in glob.glob(os.path.join(opt.music_dir, "*.wav")):
            # create temp folder (or use the cache folder if specified)
            if opt.cache_features:
                songname = os.path.splitext(os.path.basename(wav_file))[0]
                save_dir = os.path.join(opt.feature_cache_dir, songname)
                Path(save_dir).mkdir(parents=True, exist_ok=True)
                dirname = save_dir
            else:
                temp_dir = TemporaryDirectory()
                temp_dir_list.append(temp_dir)
                dirname = temp_dir.name
            # slice the audio file
            print(f"Slicing {wav_file}")
            slice_audio(wav_file, 2.5, 5.0, dirname)
            
            #-------------JK Edits-------------
            print(f"Transcribing lyrics for {wav_file}")
            whisper_result = whisper_model.transcribe(wav_file, word_timestamps=True)
            lyric_segments = []
            for seg in whisper_result["segments"]:
                emb = lyric_encoder.encode(seg["text"].strip())
                lyric_segments.append({
                    "start": seg["start"],
                    "end": seg["end"],
                    "embedding": emb
                })
            #----------------------------------
           
           
            file_list = sorted(glob.glob(f"{dirname}/*.wav"), key=stringintkey)
            # randomly sample a chunk of length at most sample_size
            rand_idx = random.randint(0, len(file_list) - sample_size)
            cond_list = []
            # generate juke representations
            print(f"Computing features for {wav_file}")
            for idx, file in enumerate(tqdm(file_list)):
                # if not caching then only calculate for the interested range
                if (not opt.cache_features) and (not (rand_idx <= idx < rand_idx + sample_size)):
                    continue
                # audio = jukemirlib.load_audio(file)
                # reps = jukemirlib.extract(
                #     audio, layers=[66], downsample_target_rate=30
                # )[66]
                reps, _ = feature_func(file)

                #-------------JK Edits-------------
                lyric_emb = get_lyric_emb_for_slice(lyric_segments, idx)
                lyric_tensor = torch.tensor(lyric_emb).unsqueeze(0).to(device)        # (1, 384)
                music_tensor = torch.tensor(reps).to(device)                          # (seq, 4800)
                lyric_tensor = lyric_tensor.expand(music_tensor.shape[0], -1)         # (seq, 384)
                reps = fuser(music_tensor, lyric_tensor).detach().cpu().numpy()
                #----------------------------------


                # save reps
                if opt.cache_features:
                    featurename = os.path.splitext(file)[0] + ".npy"
                    np.save(featurename, reps)
                # if in the random range, put it into the list of reps we want
                # to actually use for generation
                if rand_idx <= idx < rand_idx + sample_size:
                    cond_list.append(reps)
            cond_list = torch.from_numpy(np.array(cond_list))
            all_cond.append(cond_list)
            all_filenames.append(file_list[rand_idx : rand_idx + sample_size])

    model = EDGE(opt.feature_type, opt.checkpoint)
    model.eval()

    # directory for optionally saving the dances for eval
    fk_out = None
    if opt.save_motions:
        fk_out = opt.motion_save_dir

    print("Generating dances")
    for i in range(len(all_cond)):
        data_tuple = None, all_cond[i], all_filenames[i]
        model.render_sample(
            data_tuple, "test", opt.render_dir, render_count=-1, fk_out=fk_out, render=not opt.no_render
        )
    print("Done")
    torch.cuda.empty_cache()
    for temp_dir in temp_dir_list:
        temp_dir.cleanup()


if __name__ == "__main__":
    opt = parse_test_opt()
    test(opt)
