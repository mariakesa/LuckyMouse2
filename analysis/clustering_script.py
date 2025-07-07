import torch
import torch.nn as nn
import numpy as np
import pickle
from umap import UMAP
from sklearn.cluster import KMeans
import plotly.express as px
from tqdm import tqdm
import plotly.io as pio

# Use your GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pio.renderers.default = "browser"

# ---- Model Definition ----
class AttentionNullModel(nn.Module):
    def __init__(self, vit_dim, neuron_embed_dim, num_neurons, attention_dim, mlp_dim=128):
        super().__init__()
        self.neuron_embedding = nn.Embedding(num_neurons, neuron_embed_dim)
        self.input_proj = nn.Linear(vit_dim + neuron_embed_dim, attention_dim)
        self.attn = nn.MultiheadAttention(embed_dim=attention_dim, num_heads=1, batch_first=True)
        self.mlp = nn.Sequential(
            nn.LayerNorm(attention_dim),
            nn.Linear(attention_dim, mlp_dim),
            nn.ReLU(),
            nn.Linear(mlp_dim, attention_dim)
        )
        self.out = nn.Linear(attention_dim, 1)

    def forward(self, image_embedding, neuron_idx):
        neuron_emb = self.neuron_embedding(neuron_idx)
        x = torch.cat([image_embedding, neuron_emb], dim=-1)
        x = self.input_proj(x).unsqueeze(1)
        attn_out, _ = self.attn(x, x, x)
        mlp_out = self.mlp(attn_out)
        final = attn_out + mlp_out
        return self.out(final.squeeze(1))  # (B,)

# ---- Paths ----
model_path = "/home/maria/LuckyMouse2/saved_models/fold_0/model.pt"
embeddings_path = "/home/maria/LuckyMouse2/pixel_transformer_neuro/data/processed/google_vit-base-patch16-224_embeddings_softmax.pkl"

# ---- Load ViT stimulus embeddings ----
with open(embeddings_path, "rb") as f:
    vit_data = pickle.load(f)
vit_embeddings = torch.tensor(vit_data, dtype=torch.float32).to(device)  # (num_images, vit_dim)

# ---- Load Model ----
model = AttentionNullModel(
    vit_dim=1000,
    neuron_embed_dim=64,
    num_neurons=39209,
    attention_dim=128
).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# ---- Get Neuron Embeddings ----
with torch.no_grad():
    neuron_embeddings = model.neuron_embedding.weight.cpu().numpy()  # (39209, 64)

# ---- UMAP + KMeans ----
umap = UMAP(n_components=3, random_state=42)
neuron_embeddings_umap = umap.fit_transform(neuron_embeddings)

kmeans = KMeans(n_clusters=5, random_state=42)
cluster_labels = kmeans.fit_predict(neuron_embeddings_umap)

# ---- Compute Log-likelihood Proxy (Negative MSE) per neuron ----
log_likelihoods = np.zeros(model.neuron_embedding.num_embeddings)
batch_size = 512

with torch.no_grad():
    for neuron_id in tqdm(range(model.neuron_embedding.num_embeddings), desc="Computing log-likelihoods"):
        preds = []
        for i in range(0, vit_embeddings.shape[0], batch_size):
            batch_images = vit_embeddings[i:i+batch_size]
            batch_neuron_idx = torch.full((batch_images.size(0),), neuron_id, dtype=torch.long).to(device)
            out = model(batch_images, batch_neuron_idx)
            preds.append(out.cpu())

        preds = torch.cat(preds)
        mse = torch.nn.functional.mse_loss(preds, torch.zeros_like(preds), reduction='mean')
        log_likelihoods[neuron_id] = -mse.item()

# ---- Plot in UMAP space ----
fig = px.scatter_3d(
    x=neuron_embeddings_umap[:, 0],
    y=neuron_embeddings_umap[:, 1],
    z=neuron_embeddings_umap[:, 2],
    color=log_likelihoods,
    title="Neuron Embeddings (UMAP) Colored by Log-Likelihood",
    labels={"x": "UMAP-1", "y": "UMAP-2", "z": "UMAP-3"},
    color_continuous_scale="Viridis"
)
fig.update_traces(marker=dict(size=2))
fig.show()
