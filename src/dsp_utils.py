import librosa
import numpy as np
import pathlib as path

def extract_longest_word(wav_path, wrd_path):
    max_duration_samples = 0
    best_start = 0
    best_end =0
    wake_word= ""

    with open(wrd_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts)!=3: continue

            start_sample = parts[0]
            end_sample = parts[1]
            word = parts[2]

            duration = int(end_sample) - int(start_sample)

            if (duration > max_duration_samples):
                max_duration_samples = duration
                best_start = int(start_sample)
                best_end = int(end_sample)
                wake_word = word


    audio_matrix, sr = librosa.load(wav_path, sr=16000)
    # print(sr)
    isolated_word_audio = audio_matrix[best_start:best_end]
    duration_sec = max_duration_samples/sr

    return isolated_word_audio, sr, wake_word, duration_sec


def get_wrd_from_wav(wav_path: str):
    return wav_path.replace(".WAV.wav", ".WRD")