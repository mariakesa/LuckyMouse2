import numpy as np
import pandas as pd
import torch
import torch.nn as nn

def make_joined_df():

    cell_ids=np.load('/home/maria/LuckyMouse2/neuron_property_decoding/data/cell_specimen_ids_in_order.npy')
    cells_df=pd.read_csv('/home/maria/LuckyMouse2/neuron_property_decoding/data/cells_metrics_df.csv')

    # Create a DataFrame mapping data row index → cell_specimen_id
    row_map_df = pd.DataFrame({
        'cell_specimen_id': cell_ids,
        'data_row_index': np.arange(len(cell_ids))
    })

    # Merge with metrics
    merged_df = pd.merge(cells_df, row_map_df, on='cell_specimen_id', how='inner')

    # Optional: sort by row index
    merged_df = merged_df.sort_values("data_row_index").reset_index(drop=True)

    # Save or inspect
    print(f"✅ Final merged shape: {merged_df.shape}")
    print(merged_df.head())

    merged_df.to_csv('/home/maria/LuckyMouse2/neuron_property_decoding/data/cell_metrics_joined.csv')

def extract_embeddings():

    class AttentionNullModel(nn.Module):
        def __init__(self, vit_dim, neuron_embed_dim, num_neurons, attention_dim, mlp_dim=128):
            super().__init__()
            self.neuron_embedding = nn.Embedding(num_neurons, neuron_embed_dim)

            self.input_proj = nn.Linear(vit_dim + neuron_embed_dim, attention_dim)
            self.attn = nn.MultiheadAttention(
                embed_dim=attention_dim,
                num_heads=1,
                batch_first=True
            )

            # MLP block like in Transformer
            self.mlp = nn.Sequential(
                nn.LayerNorm(attention_dim),
                nn.Linear(attention_dim, mlp_dim),
                nn.ReLU(),
                nn.Linear(mlp_dim, attention_dim)
            )

            self.out = nn.Linear(attention_dim, 1)

        def forward(self, image_embedding, neuron_idx):
            # image_embedding: (B, D)
            # neuron_idx: (B,)
            neuron_emb = self.neuron_embedding(neuron_idx)  # (B, E)
            x = torch.cat([image_embedding, neuron_emb], dim=-1)  # (B, D+E)
            x = self.input_proj(x).unsqueeze(1)  # (B, 1, A)

            # Self-attention
            attn_out, _ = self.attn(x, x, x)  # (B, 1, A)

            # Transformer-style residual + MLP
            residual = attn_out
            mlp_out = self.mlp(attn_out)
            final = residual + mlp_out  # (B, 1, A)

            logits = self.out(final.squeeze(1))  # (B, A) -> (B,)
            return logits


    # Load the model
    model_path = "/home/maria/LuckyMouse2/saved_models/fold_0/model.pt"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = AttentionNullModel(
        vit_dim=1000,
        neuron_embed_dim=64,
        num_neurons=39209,
        attention_dim=128
    ).to(device)


    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Extract embeddings
    with torch.no_grad():
        neuron_embeddings = model.neuron_embedding.weight.cpu().numpy()  # shape (num_neurons, neuron_embed_dim)

    np.save('/home/maria/LuckyMouse2/neuron_property_decoding/data/neuron_embeddings.npy', neuron_embeddings)
