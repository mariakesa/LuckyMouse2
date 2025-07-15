import torch
import numpy as np
import pickle
import os
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
from scipy.stats import pearsonr
from tqdm import tqdm

# --- Paths ---
model_dir = "/home/maria/LuckyMouse2/saved_models_"
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
class AttentionNullModel(torch.nn.Module):
    def __init__(self, vit_dim, neuron_embed_dim, num_neurons, attention_dim, mlp_dim=128):
        super().__init__()
        self.neuron_embedding = torch.nn.Embedding(num_neurons, neuron_embed_dim)
        self.input_proj = torch.nn.Linear(vit_dim + neuron_embed_dim, attention_dim)
        self.attn = torch.nn.MultiheadAttention(embed_dim=attention_dim, num_heads=1, batch_first=True)
        self.mlp = torch.nn.Sequential(
            torch.nn.LayerNorm(attention_dim),
            torch.nn.Linear(attention_dim, mlp_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(mlp_dim, attention_dim)
        )
        self.out = torch.nn.Linear(attention_dim, 1)

    def forward(self, image_embedding, neuron_idx):
        neuron_emb = self.neuron_embedding(neuron_idx)
        x = torch.cat([image_embedding, neuron_emb], dim=-1)
        x = self.input_proj(x).unsqueeze(1)
        attn_out, _ = self.attn(x, x, x)
        mlp_out = self.mlp(attn_out)
        final = attn_out + mlp_out
        return self.out(final.squeeze(1))

num_folds = 10
kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)

r2_scores = []
correlations = []
sigmoid = torch.nn.Sigmoid()

for fold_idx, (train_img_idx, test_img_idx) in enumerate(kf.split(np.arange(num_images))):
    print(f"\n🧪 Evaluating Fold {fold_idx}")

    # Load model for this fold
    model_path = os.path.join(model_dir, f"fold_{fold_idx}", "model.pt")
    model = AttentionNullModel(
        vit_dim=1000,
        neuron_embed_dim=64,
        num_neurons=39209, 
        attention_dim=128
    )
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    # Predict for all neurons, only on test images
    preds = np.zeros((num_neurons, len(test_img_idx)), dtype=np.float32)

    with torch.no_grad():
        for neuron_id in tqdm(range(num_neurons), desc=f"Predicting Fold {fold_idx}"):
            test_embeddings = vit_embeddings[test_img_idx]  # (num_test_images, D)
            neuron_idx = torch.full((len(test_img_idx),), neuron_id, dtype=torch.long)
            output = model(test_embeddings, neuron_idx)
            probs = sigmoid(output).cpu().numpy().squeeze()
            preds[neuron_id] = probs  # now shape is (num_test_images,)

    # Compare with empirical probs on test images
    y_true = empirical_probs[:, test_img_idx].flatten()
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
