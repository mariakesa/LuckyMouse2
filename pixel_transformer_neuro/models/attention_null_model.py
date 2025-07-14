import torch
import torch.nn as nn

class AttentionNullModel(nn.Module):
    def __init__(self, vit_dim, neuron_embed_dim, num_neurons, attention_dim, mlp_dim=128):
        super().__init__()
        self.neuron_embedding = nn.Embedding(num_neurons, neuron_embed_dim)

        self.input_proj = nn.Linear(vit_dim + neuron_embed_dim, attention_dim)

        self.attn = nn.MultiheadAttention(
            embed_dim=1,
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
        x = torch.randn(64, 128)      # (B, 512)
        x = x.unsqueeze(-1)
        # Self-attention
        attn_out, _ = self.attn(x, x, x)  # (B, 1, A)

        # Transformer-style residual + MLP
        residual = attn_out
        mlp_out = self.mlp(attn_out)
        final = residual + mlp_out  # (B, 1, A)

        logits = self.out(final.squeeze(1))  # (B, A) -> (B,)
        return logits

import torch
import torch.nn as nn

import torch
import torch.nn as nn

import torch
import torch.nn as nn

class AttentionNullModel(nn.Module):
    def __init__(self, vit_dim, neuron_embed_dim, num_neurons,
                 attention_dim, mlp_dim=128, num_tokens=30, num_heads=4):
        super().__init__()
        self.num_tokens = num_tokens
        self.neuron_embedding = nn.Embedding(num_neurons, neuron_embed_dim)

        input_dim = vit_dim + neuron_embed_dim

        # 30 learnable projections of the full input vector (each into attention_dim)
        self.token_projections = nn.ModuleList([
            nn.Linear(input_dim, attention_dim) for _ in range(num_tokens)
        ])

        # Positional embeddings for the pseudo-token sequence
        self.pos_emb = nn.Parameter(torch.randn(1, num_tokens, attention_dim))

        # Multi-head self-attention over the projected tokens
        self.attn = nn.MultiheadAttention(
            embed_dim=attention_dim,
            num_heads=num_heads,
            batch_first=True
        )

        # Transformer-style MLP block
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
        neuron_emb = self.neuron_embedding(neuron_idx)              # (B, E)
        x = torch.cat([image_embedding, neuron_emb], dim=-1)        # (B, D+E)

        # Generate sequence of projected tokens
        tokens = torch.stack([proj(x) for proj in self.token_projections], dim=1)  # (B, T, A)
        tokens = tokens + self.pos_emb                                           # (B, T, A)

        # Self-attention across tokens
        attn_out, _ = self.attn(tokens, tokens, tokens)  # (B, T, A)

        # MLP + residual
        x = attn_out + self.mlp(attn_out)  # (B, T, A)

        # Pool across tokens (mean)
        pooled = x.mean(dim=1)  # (B, A)

        # Output prediction
        logits = self.out(pooled).squeeze(-1)  # (B,)

        return 
    
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

