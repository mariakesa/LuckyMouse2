import torch
import numpy as np
import pickle
import os
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
from scipy.stats import pearsonr
from tqdm import tqdm
import torch.nn as nn

# --- Paths ---
model_dir = "/home/maria/LuckyMouse2/saved_models"
embedding_path = "/home/maria/LuckyMouse2/pixel_transformer_neuro/data/processed/google_vit-base-patch16-224_embeddings_softmax.pkl"
response_path = "/home/maria/LuckyMouse2/pixel_transformer_neuro/data/processed/hybrid_neural_responses_reduced.npy"

# --- Load data ---
with open(embedding_path, "rb") as f:
    vit_embeddings = pickle.load(f)['natural_scenes']
vit_embeddings = torch.tensor(vit_embeddings, dtype=torch.float32)

empirical_probs = np.load(response_path) / 50  # shape: (num_neurons, num_images)

num_neurons, num_images = empirical_probs.shape
num_folds = 10

# --- Model class ---
class AttentionNullModel(nn.Module):
    def __init__(self, vit_dim, neuron_embed_dim, num_neurons, mlp_dim=128):
        super().__init__()
        assert neuron_embed_dim == vit_dim, "To allow elementwise multiplication, neuron_embed_dim must match vit_dim"

        # Each neuron has a learned modulation vector (same dim as image embedding)
        self.neuron_embedding = nn.Embedding(num_neurons, neuron_embed_dim)

        # MLP to predict from modulated stimulus
        self.mlp = nn.Sequential(
            nn.LayerNorm(vit_dim),
            nn.Linear(vit_dim, mlp_dim),
            nn.ReLU(),
            nn.Linear(mlp_dim, 1)
        )

    def forward(self, image_embedding, neuron_idx):
        """
        image_embedding: (B, D) - e.g. ViT softmax vector
        neuron_idx:      (B,)   - neuron indices
        """
        neuron_vector = self.neuron_embedding(neuron_idx)      # (B, D)
        modulated = image_embedding + torch.sigmoid(neuron_vector)               # (B, D) - elementwise modulation
        logits = self.mlp(modulated).squeeze(-1)                  # (B,)
        return logits


num_folds = 10
kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)

r2_scores = []
correlations = []
sigmoid = torch.nn.Sigmoid()

# === CHANGES TO SCRIPT: Tail-specific R² computation ===

# Inside the KFold loop
r2_scores_all = []
r2_scores_tail = []
correlations = []

for fold_idx, (train_img_idx, test_img_idx) in enumerate(kf.split(np.arange(num_images))):
    print(f"\n🧪 Evaluating Fold {fold_idx}")

    # Load model
    model_path = os.path.join(model_dir, f"fold_{fold_idx}", "model.pt")
    model = AttentionNullModel(
        vit_dim=1000,
        neuron_embed_dim=1000,
        num_neurons=num_neurons
    )
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    # Predict for all neurons on test set
    preds = np.zeros((num_neurons, len(test_img_idx)), dtype=np.float32)

    with torch.no_grad():
        for neuron_id in tqdm(range(num_neurons), desc=f"Predicting Fold {fold_idx}"):
            test_embeddings = vit_embeddings[test_img_idx]
            neuron_idx = torch.full((len(test_img_idx),), neuron_id, dtype=torch.long)
            output = model(test_embeddings, neuron_idx)
            probs = sigmoid(output).cpu().numpy()
            print(probs)
            preds[neuron_id] = probs

    # Flatten predictions and true values
    y_true = empirical_probs[:, test_img_idx].flatten()
    y_pred = preds.flatten()
    mask = np.isfinite(y_true) & np.isfinite(y_pred)

    # Full R² and correlation
    r2_all = r2_score(y_true[mask], y_pred[mask])
    corr, _ = pearsonr(y_true[mask], y_pred[mask])
    r2_scores_all.append(r2_all)
    correlations.append(corr)

    # R² on tail (y_true < 0.2)
    tail_mask = (y_true < 0.15) & mask
    if np.sum(tail_mask) > 0:
        r2_tail = r2_score(y_true[tail_mask], y_pred[tail_mask])
    else:
        r2_tail = np.nan  # not enough tail points
    r2_scores_tail.append(r2_tail)

# --- Summary ---
print("\n📊 Cross-validated Results:")
print(f"Mean R² (all):     {np.nanmean(r2_scores_all):.4f} ± {np.nanstd(r2_scores_all):.4f}")
print(f"Mean R² (tail<0.2): {np.nanmean(r2_scores_tail):.4f} ± {np.nanstd(r2_scores_tail):.4f}")
print(f"Mean Pearson r:     {np.nanmean(correlations):.4f} ± {np.nanstd(correlations):.4f}")
