import pickle
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

def inspect_ubm():
    model_path = Path(r"results\models\ubm_model.pkl")

    with open(model_path, 'rb') as f:
        ubm = pickle.load(f)

    print(f"Total Gaussians: {ubm.n_components}")
    print(f"Means Matrix Shape:         {ubm.means_.shape}")
    print(f"Covariance Matrix Shape:    {ubm.covariances_.shape}")
    print(f"Did the math converge?      {ubm.converged_}")
    print(f"Iterations to converge:     {ubm.n_iter_}")
    
    weights = ubm.weights_
    print(f"\nMax weight assigned:        {np.max(weights):.4f}")
    print(f"Min weight assigned:        {np.min(weights):.4f}")
    
    plt.figure(figsize=(10, 4))
    
    plt.bar(range(ubm.n_components), weights, color='teal', edgecolor='black')
    
    plt.title("UBM Gaussian Mixture Weights\n(How much 'importance' each cluster holds)")
    plt.xlabel("Gaussian Component Index (0 to 63)")
    plt.ylabel("Weight (Probability)")
    
    plt.axhline(y=1/64, color='red', linestyle='--', alpha=0.7, label='Average Weight')
    
    plt.legend()
    plt.tight_layout()
    plt.show()


def main() -> None:
    inspect_ubm()

if __name__ == "__main__":
    main()