import torch
import torch.nn as nn

class AttentionNullModel(nn.Module):
    def __init__(self, vit_dim, neuron_embed_dim, num_neurons, attention_dim):
        super().__init__()
        self.neuron_embedding = nn.Embedding(num_neurons, neuron_embed_dim)

        self.input_proj = nn.Linear(vit_dim + neuron_embed_dim, attention_dim)

        self.attn = nn.MultiheadAttention(embed_dim=attention_dim, num_heads=1, batch_first=True)

        self.out = nn.Sequential(
            nn.Linear(attention_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, image_embedding, neuron_idx):
        # image_embedding: (B, D)
        # neuron_idx: (B,)

        neuron_emb = self.neuron_embedding(neuron_idx)  # (B, E)
        x = torch.cat([image_embedding, neuron_emb], dim=-1)  # (B, D+E)
        x = self.input_proj(x).unsqueeze(1)  # (B, 1, A)

        # Use x as both query, key, and value (null self-attention)
        attn_out, _ = self.attn(x, x, x)  # (B, 1, A)
        logits = self.out(attn_out.squeeze(1))  # (B, 1) -> (B,)
        return logits
