import pytorch_lightning as pl
import torch
import torch.nn as nn

from config import config

WINDOW_SIZE = config.PARAMS["WINDOW_SIZE"]
num_exercise = config.PARAMS["num_exercise"]
num_augment = config.PARAMS["num_augment"]
num_classes_ = config.PARAMS["TOT_CLASSES"]


device = "cuda" if torch.cuda.is_available() else "cpu"


class PositionalEmbedding(nn.Module):
    def __init__(self, sequence_length, embed_size):
        super().__init__()
        self.sequence_length = sequence_length
        self.output_dim = 1
        # The inputs are of shape: `(batch_size, frames, num_features)
        self.positions = torch.tensor(
            [
                int(val)
                for val in torch.range(start=0, end=self.sequence_length - 1, step=1)
            ]
        ).to(device)
        self.position_embeddings = nn.Embedding(self.sequence_length, self.output_dim)

    def forward(self, inputs):
        return inputs + self.position_embeddings(self.positions)


class TRANSFORMERENCODER(nn.Module):
    """
    transformer block by pytorch lib from the paper "Attention is all you need"
    adapted to ViT 1D algorithm (2 blocks, 2 heads)
    """

    def __init__(self, embed_dim, dense_dim, num_heads):
        super().__init__()

        self.sequence_length = WINDOW_SIZE * num_exercise
        self.embed_dim = embed_dim
        self.dense_dim = dense_dim
        self.num_heads = num_heads
        self.attention = nn.MultiheadAttention(
            embed_dim,
            num_heads,
            dropout=0.1,
            bias=True,
            add_bias_kv=False,
            add_zero_attn=False,
            kdim=None,
            vdim=None,  # batch_first=False,
        )

        self.layernorm_1 = nn.LayerNorm(embed_dim)
        self.layernorm_2 = nn.LayerNorm(embed_dim)

        self.query_h = nn.Linear(embed_dim, embed_dim, bias=False)
        self.key_h = nn.Linear(embed_dim, embed_dim, bias=False)
        self.value_h = nn.Linear(embed_dim, embed_dim, bias=False)

        self.fnn = nn.Sequential(
            nn.Linear(embed_dim, dense_dim),
            nn.GELU(),
            nn.Linear(dense_dim, embed_dim),
            nn.Dropout(0.0),
        )

    def forward(self, inputs):
        Q, K, V = self.query_h(inputs), self.key_h(inputs), self.value_h(inputs)
        attention_output, attention_output_weights = self.attention(Q, K, V)
        proj_input = self.layernorm_1(inputs + attention_output)
        proj_output = self.fnn(proj_input)
        return self.layernorm_2(proj_input + proj_output)


class TRANSFORMER(pl.LightningModule):
    def __init__(self, embed_dim, dense_dim, num_heads, output_size):
        super().__init__()
        self.save_hyperparameters()
        self.sequence_length = WINDOW_SIZE * num_exercise
        self.embed_dim = embed_dim
        self.dense_dim = dense_dim
        self.num_heads = num_heads
        self.classes = output_size
        blocks = 2

        self.positionalembedding = PositionalEmbedding(
            self.sequence_length, self.embed_dim
        )
        self.block_list = [
            TRANSFORMERENCODER(self.embed_dim, self.dense_dim, self.num_heads)
            for _ in range(blocks)
        ]
        self.layers = nn.ModuleList(self.block_list)

        self.fc1 = nn.Linear(self.embed_dim, self.classes)

    def forward(self, x):
        x = self.positionalembedding(x)
        for layer in self.layers:
            out = layer(x)
        out = self.fc1(out)[:, -1]
        return out
