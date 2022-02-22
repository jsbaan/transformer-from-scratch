import unittest
import math

import torch
from torch import nn
from torch.nn.init import xavier_uniform_

from multi_head_attention import MultiHeadAttention
from positional_encodings import SinusoidEncoding
from vocabulary import Vocabulary


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        embedding: torch.nn.Embedding,
        hidden_dim: int,
        ff_dim: int,
        num_heads: int,
        num_layers: int,
        dropout_p: float,
    ):
        super().__init__()
        self.embed = embedding
        self.hidden_dim = hidden_dim
        self.positional_encoding = SinusoidEncoding(hidden_dim, max_len=5000)
        self.dropout = nn.Dropout(p=dropout_p)
        self.encoder_blocks = nn.ModuleList(
            [EncoderBlock(hidden_dim, ff_dim, num_heads, dropout_p) for _ in range(num_layers)]
        )

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)

    def forward(self, input_ids: torch.Tensor, mask: torch.BoolTensor = None):  # (n_batches, sequence_length)
        # (n_batches, sequence_length, hidden_dim)
        x = self.embed(input_ids) * math.sqrt(self.hidden_dim)
        x = self.positional_encoding(x)
        x = self.dropout(x)
        for encoder_block in self.encoder_blocks:
            x = encoder_block.forward(x, mask=mask)  # (n_batches, sequence_length, hidden_dim)
        return x


class EncoderBlock(nn.Module):
    def __init__(self, hidden_dim: int, ff_dim: int, num_heads: int, dropout_p: float):
        super().__init__()
        self.self_mha = MultiHeadAttention(hidden_dim, num_heads)
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, hidden_dim),
        )

        self.dropout1 = nn.Dropout(p=dropout_p)
        self.dropout2 = nn.Dropout(p=dropout_p)
        self.layer_norm1 = nn.LayerNorm(hidden_dim)
        self.layer_norm2 = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor, mask: torch.BoolTensor = None):
        # (n_batches, sequence_length, hidden_dim)
        output = self.dropout1(self.self_mha.forward(x, mask=mask))
        x = self.layer_norm1(x + output)

        output = self.dropout2(self.feed_forward(x))
        x = self.layer_norm2(x + output)
        return x


class TestTransformerEncoder(unittest.TestCase):
    def test_transformer_encoder_single_sequence_batch(self):
        # Create vocabulary and special token indices given a dummy corpus
        batch = ["Hello my name is Joris and I was born with the name Joris."]
        en_vocab = Vocabulary(batch)
        en_vocab_size = len(en_vocab.token2index.items())
        with torch.no_grad():
            # Initialize a transformer encoder (qkv_dim is automatically set to hidden_dim // num_heads)
            encoder = TransformerEncoder(
                embedding=torch.nn.Embedding(en_vocab_size, 512),
                hidden_dim=512,
                ff_dim=2048,
                num_heads=8,
                num_layers=2,
                dropout_p=0.1
            )
            encoder._reset_parameters()
            encoder.eval()
            # Construct input tensor
            input_batch = torch.IntTensor(
                en_vocab.batch_encode(batch, add_special_tokens=False)
            )

            output = encoder.forward(input_batch)
            self.assertEqual(output.shape, (1, 14, 512))

    def test_transformer_encoder_multi_sequence_batch(self):
        # Create vocabulary and special token indices given a dummy batch
        batch = [
            "Hello my name is Joris and I was born with the name Joris.",
            "A shorter sequence in the batch",
        ]
        en_vocab = Vocabulary(batch)
        en_vocab_size = len(en_vocab.token2index.items())

        # Initialize a transformer encoder (qkv_dim is automatically set to hidden_dim // num_heads)
        with torch.no_grad():
            encoder = TransformerEncoder(
                embedding=torch.nn.Embedding(en_vocab_size, 512),
                hidden_dim=512,
                ff_dim=2048,
                num_heads=8,
                num_layers=2,
                dropout_p=0.1
            )
            encoder.eval()
            input_batch = torch.IntTensor(
                en_vocab.batch_encode(batch, add_special_tokens=False, padding=True)
            )
            src_mask = input_batch == en_vocab.token2index[en_vocab.PAD]

            output = encoder.forward(input_batch, mask=src_mask)
            self.assertEqual(output.shape, (2, 14, 512))


if __name__ == "__main__":
    unittest.main()
