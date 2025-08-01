import pickle
import numpy as np
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm

# ─── Load Data ───────────────────────────────────────────────────────────────
dataset_path_dict = {
    "embeddings": "/home/maria/Documents/HuggingMouseData/MouseViTEmbeddings/google_vit-base-patch16-224_embeddings_logits.pkl",
    "neural": "/home/maria/LuckyMouse2/pixel_transformer_neuro/data/processed/hybrid_neural_responses.npy"
}

with open(dataset_path_dict['embeddings'], "rb") as f:
    embeddings_raw = pickle.load(f)
embeddings = embeddings_raw['natural_scenes']  # shape: (118, 1000)
print("Full embedding shape:", embeddings.shape)

neural_data = np.load(dataset_path_dict["neural"])  # shape: (neurons, 5900)
print("Neural data shape:", neural_data.shape)

# ─── Construct Trial-wise Design Matrix ──────────────────────────────────────
n_images = embeddings.shape[0]        # 118
n_trials = 50
n_total = n_images * n_trials         # 5900

X_all = np.repeat(embeddings, n_trials, axis=0)  # shape: (5900, 1000)
n_neurons = neural_data.shape[0]
embedding_dim = X_all.shape[1]

# ─── Fit Model for Each Neuron ───────────────────────────────────────────────
representation_vectors = np.zeros((n_neurons, embedding_dim))
failures = []

for i in tqdm(range(n_neurons), desc="Fitting neurons"):
    y = neural_data[i]
    try:
        model = LogisticRegression(solver="liblinear", max_iter=1000)
        model.fit(X_all, y)
        representation_vectors[i] = model.coef_[0]
    except Exception as e:
        print(f"Neuron {i}: Error - {e}")
        representation_vectors[i] = np.nan
        failures.append(i)

# ─── Save to .npy ────────────────────────────────────────────────────────────
output_path = "/home/maria/LuckyMouse2/pixel_transformer_neuro/data/processed/logreg_representation_vectors.npy"
np.save(output_path, representation_vectors)
print(f"Saved representation vectors to {output_path}")

if failures:
    print(f"Failed to fit {len(failures)} neurons: {failures}")
