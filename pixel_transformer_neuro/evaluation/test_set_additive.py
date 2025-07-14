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

# --- Evaluation loop ---
kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)
r2_scores = []
correlations = []
sigmoid = torch.nn.Sigmoid()

vit_embeddings = vit_embeddings.to("cpu")

for fold_idx, (train_idx, test_idx) in enumerate(kf.split(np.arange(num_neurons))):
    print(f"\n🧪 Evaluating Fold {fold_idx}")

    model_path = os.path.join(model_dir, f"fold_{fold_idx}", "model.pt")
    model = AttentionNullModel(
        vit_dim=1000,
        neuron_embed_dim=1000,
        num_neurons=num_neurons
    )
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    preds = np.zeros((len(test_idx), num_images), dtype=np.float32)

    with torch.no_grad():
        for j, neuron_id in enumerate(tqdm(test_idx, desc=f"Predicting Fold {fold_idx}")):
            probs = []
            for i in range(0, num_images, 256):
                image_batch = vit_embeddings[i:i+256]
                neuron_batch = torch.full((image_batch.shape[0],), neuron_id, dtype=torch.long)
                out = model(image_batch, neuron_batch)
                p = sigmoid(out).squeeze().cpu().numpy()
                probs.append(p)
            preds[j] = np.concatenate(probs)

    # True values
    y_true = empirical_probs[test_idx].flatten()
    y_pred = preds.flatten()
    mask = np.isfinite(y_true) & np.isfinite(y_pred)

    r2 = r2_score(y_true[mask], y_pred[mask])
    corr, _ = pearsonr(y_true[mask], y_pred[mask])
    r2_scores.append(r2)
    correlations.append(corr)

# --- Summary ---
print("\n📊 Cross-validated Results:")
print(f"Mean R²:  {np.mean(r2_scores):.4f} ± {np.std(r2_scores):.4f}")
print(f"Mean r:   {np.mean(correlations):.4f} ± {np.std(correlations):.4f}")
