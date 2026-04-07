import numpy as np
import librosa
import matplotlib.pyplot as plt

def extract_mfcc_features(audio, sr=16000, n_mfcc=20):
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc, n_fft=400, hop_length= 160)

    d1 = librosa.feature.delta(mfcc)
    d2 = librosa.feature.delta(mfcc, order=2)

    features = np.vstack([mfcc, d1, d2]).T
    feats = (features- np.mean(features, axis=0))/(np.std(features, axis=0) + 1e-8)

    return feats


if __name__== "__main__":
    sr = 16000
    n_mfcc = 20
    duration = 1.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    audio = 0.1 * np.sin(2 * np.pi * 440 * t)

    feats = extract_mfcc_features(audio, sr=sr, n_mfcc=n_mfcc)
    print(f"Extracted features shape: {feats.shape} (frames x {3 * n_mfcc})")
    print(f"Mean≈0: {np.allclose(np.mean(feats, axis=0), 0, atol=1e-5)}  Std≈1: {np.allclose(np.std(feats, axis=0), 1, atol=1e-3)}")

    #original signal
    duration_sec = len(audio) / sr if sr else 0
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), constrained_layout=True)

    # Waveform
    axes[0].plot(t, audio, linewidth=1.0)
    axes[0].set_title("Waveform")
    axes[0].set_xlabel("Time [s]")
    axes[0].set_ylabel("Amplitude")

    # Features heatmap (MFCC + deltas), normalized
    im = axes[1].imshow(
        feats.T,
        aspect="auto",
        origin="lower",
        extent=[0, duration_sec, 0, feats.shape[1]],
        cmap="magma",
        interpolation="nearest",
    )
    axes[1].set_title("MFCC + delta + delta² (normalized)")
    axes[1].set_xlabel("Time [s]")
    axes[1].set_ylabel("Feature index")
    fig.colorbar(im, ax=axes[1], label="Value")

    plt.show()