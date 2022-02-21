import unittest
from typing import Optional

import torch
from torch import nn

from multi_head_attention import MultiHeadAttention
from positional_encodings import SinusoidEncoding
from vocabulary import Vocabulary


class TransformerEncoder(nn.Module):
    def __init__(
        self, embedding: torch.nn.Embedding, hidden_dim: int, ff_dim: int, num_heads: int, num_layers: int
    ):
        super().__init__()
        self.embed = embedding
        self.positional_encoding = SinusoidEncoding(hidden_dim, max_len=5000)
        self.encoder_blocks = nn.ModuleList(
            [EncoderBlock(hidden_dim, ff_dim, num_heads) for _ in range(num_layers)]
        )

    def forward(self, input_ids: torch.Tensor):
        x = self.embed(input_ids)  # (n_batches, sequence_length) to (n_batches, sequence_length, hidden_dim)
        x = self.positional_encoding(x)
        for encoder_block in self.encoder_blocks:
            x = encoder_block.forward(x)  # (n_batches, sequence_length, hidden_dim)
        return x


class EncoderBlock(nn.Module):
    def __init__(self, hidden_dim: int, ff_dim: int, num_heads: int):
        super().__init__()
        self.multi_head_attention = MultiHeadAttention(hidden_dim, num_heads)
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_dim, ff_dim), nn.ReLU(), nn.Linear(ff_dim, hidden_dim)
        )
        # TODO verify this, I don't fully understand layer norm
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor):
        mha_output = self.multi_head_attention.forward(x)  # (n_batches, sequence_length, hidden_dim)
        x = self.layer_norm(x + mha_output)

        ff_output = self.feed_forward(x)  # (n_batches, sequence_length, hidden_dim)
        x = self.layer_norm(x + ff_output)
        return x


class TestTransformerEncoder(unittest.TestCase):
    def test_transformer_encoder_single_sequence_batch(self):
        # Create vocabulary and special token indices given a dummy corpus
        batch = ["Hello my name is Joris and I was born with the name Joris."]
        en_vocab = Vocabulary(batch)
        en_vocab_size = len(en_vocab.token2index.items())

        # Initialize a transformer encoder (qkv_dim is automatically set to hidden_dim // num_heads)
        encoder = TransformerEncoder(
            embedding=torch.nn.Embedding(en_vocab_size, 512),
            hidden_dim=512,
            ff_dim=2048,
            num_heads=8,
            num_layers=2,
        )
        # Construct input tensor
        input_batch = torch.IntTensor(en_vocab.batch_encode(batch, add_special_tokens=False))

        output = encoder.forward(input_batch)
        self.assertEqual(output.shape, (1, 14, 512))

    def test_transformer_encoder_multi_sequence_batch(self):
        # Create vocabulary and special token indices given a dummy batch
        batch = ["Hello my name is Joris and I was born with the name Joris.", "A shorter sequence in the batch"]
        en_vocab = Vocabulary(batch)
        en_vocab_size = len(en_vocab.token2index.items())

        # Initialize a transformer encoder (qkv_dim is automatically set to hidden_dim // num_heads)
        encoder = TransformerEncoder(
            embedding=torch.nn.Embedding(en_vocab_size, 512),
            hidden_dim=512,
            ff_dim=2048,
            num_heads=8,
            num_layers=2,
        )
        input_batch = torch.IntTensor(en_vocab.batch_encode(batch, add_special_tokens=False, padding=True))

        output = encoder.forward(input_batch)
        self.assertEqual(output.shape, (2, 14, 512))


if __name__ == "__main__":
    unittest.main()
