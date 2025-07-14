import torch
import torch.nn as nn

class PixelAttentionModel(nn.Module):
    def __init__(self, vit_dim, num_tokens, attention_dim, num_neurons, neuron_embed_dim, mlp_dim=128):
        super().__init__()
        self.num_tokens = num_tokens
        self.token_dim = vit_dim // num_tokens
        assert vit_dim % num_tokens == 0, "vit_dim must be divisible by num_tokens"

        self.neuron_embedding = nn.Embedding(num_neurons, neuron_embed_dim)
        
        # Project each token + neuron embedding to attention_dim
        self.token_proj = nn.Linear(self.token_dim + neuron_embed_dim, attention_dim)

        self.attn = nn.MultiheadAttention(
            embed_dim=attention_dim,
            num_heads=4,
            batch_first=True
        )

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
        B = image_embedding.size(0)

        # Reshape image_embedding to sequence of pseudo-pixels
        x = image_embedding.view(B, self.num_tokens, self.token_dim)  # (B, L, D')

        # Get neuron embedding and expand to sequence
        neuron_emb = self.neuron_embedding(neuron_idx)  # (B, E)
        neuron_emb_seq = neuron_emb.unsqueeze(1).repeat(1, self.num_tokens, 1)  # (B, L, E)

        # Concatenate token + neuron conditioning
        x = torch.cat([x, neuron_emb_seq], dim=-1)  # (B, L, D'+E)
        x = self.token_proj(x)  # (B, L, A)

        x = torch.randn(64, 128)      # (B, 512)
        x_seq = x.unsqueeze(-1)
        # Self-attention
        attn_out, _ = self.attn(x, x, x)  # (B, L, A)

        # Residual + MLP
        x = attn_out + self.mlp(attn_out)  # (B, L, A)

        # Pooling across the sequence: mean or CLS-style
        poo
