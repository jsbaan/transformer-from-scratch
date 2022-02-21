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
        vocab_size: int,
        dropout_p: float,
    ):
        super().__init__()
        self.embed = embedding
        self.positional_encoding = SinusoidEncoding(hidden_dim)
        self.dropout = nn.Dropout(p=0.1)
        self.decoder_blocks = nn.ModuleList(
            [
                TransformerDecoderBlock(hidden_dim, ff_dim, num_heads, dropout_p)
                for _ in range(num_layers)
            ]
        )
        # TODO use transpose embed weights for this
        self.output_layer = nn.Linear(hidden_dim, vocab_size)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, input_tokens: torch.Tensor, encoder_hidden_states: torch.Tensor):
        """
        Output=dim(n_batches, seq_len, vocab_size). The last token index contains the next token predictive distribution
        """
        # (n_batches, sequence_length, hidden_dim)
        x = self.embed(input_tokens)
        x = self.positional_encoding(x)
        x = self.dropout(x)

        for decoder_block in self.decoder_blocks:
            x = decoder_block(x, encoder_hidden_states)

        # (n_batches, sequence_length, vocab_size)
        logits = self.output_layer(x)
        output = self.softmax(logits)
        return output


class TransformerDecoderBlock(nn.Module):
    def __init__(self, hidden_dim: int, ff_dim: int, num_heads: int, dropout_p: float):
        super().__init__()
        self.cross_mha = MultiHeadAttention(hidden_dim, num_heads)
        self.dropout_cross_mha = nn.Dropout(p=dropout_p)

        self.self_mha = MultiHeadAttention(hidden_dim, num_heads)
        self.dropout_self_mha = nn.Dropout(p=dropout_p)

        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_dim, ff_dim),
            nn.ReLU(inplace=True),
            nn.Linear(ff_dim, hidden_dim),
        )
        self.dropout_ff = nn.Dropout(p=dropout_p)
        # TODO verify this api call, I don't fully understand layer norm
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor, encoder_hidden_states: torch.Tensor):
        # Future-masked self attention TODO add masking
        output = self.dropout_self_mha(
            self.self_mha.forward(x)
        )  # (n_batches, sequence_length, hidden_dim)
        residual_output = self.layer_norm(x + output)

        # Cross or encoder-decoder attention
        output = self.dropout_cross_mha(
            self.cross_mha.forward(residual_output, encoder_hidden_states)
        )
        residual_output = self.layer_norm(residual_output + output)

        # Feed forward layers
        output = self.dropout_ff(self.feed_forward(x))
        residual_output = self.layer_norm(residual_output + output)
        return residual_output


class TestTransformerDecoder(unittest.TestCase):
    def test_transformer_decoder(self):
        # TODO implement
        raise NotImplementedError()


if __name__ == "__main__":
    unittest.main()
