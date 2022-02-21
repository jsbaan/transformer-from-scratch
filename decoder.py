import unittest
from typing import Optional

import torch
from torch import nn

from vocabulary import Vocabulary
from multi_head_attention import MultiHeadAttention
from positional_encodings import SinusoidEncoding


class TransformerDecoder(nn.Module):
    def __init__(
        self,
        embedding: torch.nn.Embedding,
        hidden_dim: int,
        ff_dim: int,
        num_heads: int,
        num_layers: int,
        vocab_size: int
    ):
        super().__init__()
        self.embed = embedding
        self.positional_encoding = SinusoidEncoding(hidden_dim)
        self.decoder_blocks = nn.ModuleList(
            [TransformerDecoderBlock(hidden_dim, ff_dim, num_heads) for _ in range(num_layers)]
        )
        # TODO use transpose embed weights for this
        self.output_layer = nn.Linear(hidden_dim, vocab_size)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, input_tokens: torch.Tensor, encoder_hidden_states: torch.Tensor):
        """
        Output=dim(n_batches, seq_len, vocab_size). The last token index contains the next token predictive distribution
        """
        x = self.embed(input_tokens)
        x = self.positional_encoding(x)
        for decoder_block in self.decoder_blocks:
            x = decoder_block(x, encoder_hidden_states)  # (n_batches, sequence_length, hidden_dim)
        logits = self.output_layer(x)  # (n_batches, sequence_length, vocab_size)
        output = self.softmax(logits)
        return output


class TransformerDecoderBlock(nn.Module):
    def __init__(self, hidden_dim: int, ff_dim: int, num_heads: int):
        super().__init__()
        self.cross_mha = MultiHeadAttention(hidden_dim, num_heads)
        self.self_mha = MultiHeadAttention(hidden_dim, num_heads)
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_dim, ff_dim), nn.ReLU(), nn.Linear(ff_dim, hidden_dim)
        )
        # TODO verify this, I don't fully understand layer norm
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor, encoder_hidden_states: torch.Tensor):
        # TODO add masked self attention here
        mha_output = self.cross_mhamulti_head_attention.forward(x)  # (n_batches, sequence_length, hidden_dim)
        x = self.layer_norm(x + mha_output)

        ff_output = self.feed_forward(x)  # (n_batches, sequence_length, hidden_dim)
        x = self.layer_norm(x + ff_output)

        # TODO add cross-attention here
        return x


class TestTransformerDecoder(unittest.TestCase):
    def test_transformer_decoder(self):
        # TODO implement
        raise NotImplementedError()


if __name__ == "__main__":
    unittest.main()
