import librosa
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from src.dsp_utils import extract_longest_word

def test_word_extraction():
    wav_path = Path(r"data\raw_timit\data\TEST\DR8\FCMH1\SI1493.WAV.wav")
    
    wrd_path = wav_path.with_suffix('').with_suffix('.WRD')
    
    print(f"Testing Audio: {wav_path.name}")
    print(f"Testing Trans: {wrd_path.name}")
    
    isolated_audio, sr, wake_word, duration = extract_longest_word(wav_path, wrd_path)
    
    print(f"\n--- Extraction Results ---")
    print(f"Wake Word Found: '{wake_word}'")
    print(f"Duration:        {duration:.3f} seconds")
    print(f"Matrix Shape:    {isolated_audio.shape}")
    
    full_audio, sr = librosa.load(wav_path, sr=sr)
    
    plt.figure(figsize=(10, 4))
    
    time_full = np.linspace(0, len(full_audio)/sr, len(full_audio))
    
    plt.plot(time_full, full_audio, color='lightgray', label='Full SI1493 Sentence')
    
    from scipy.signal import correlate
    correlation = correlate(full_audio, isolated_audio, mode='valid')
    start_idx = np.argmax(correlation)
    
    time_slice = np.linspace(start_idx/sr, (start_idx+len(isolated_audio))/sr, len(isolated_audio))
    
    plt.plot(time_slice, isolated_audio, color='blue', label=f'Extracted: "{wake_word}"')
    
    plt.title(f"DSP Slicing Verification - Wake Word: '{wake_word}'")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    test_word_extraction()