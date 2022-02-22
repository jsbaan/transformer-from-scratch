import math
import unittest
import random
from typing import Optional

import numpy as np
import torch
from torch import nn
from torch.nn.init import xavier_uniform_

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

        self.hidden_dim = hidden_dim
        self.embed = embedding
        self.positional_encoding = SinusoidEncoding(hidden_dim)
        self.dropout = nn.Dropout(p=0.1)
        self.decoder_blocks = nn.ModuleList(
            [
                TransformerDecoderBlock(hidden_dim, ff_dim, num_heads, dropout_p)
                for _ in range(num_layers)
            ]
        )
        self.output_layer = nn.Linear(hidden_dim, vocab_size, bias=False)
        # Note: a linear layer multiplies the input with a transpose of the weight matrix, so no need to do that here.
        self.output_layer.weight = nn.Parameter(self.embed.weight)
        self.softmax = nn.Softmax(dim=-1)

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)

    def forward(
        self,
        input_tokens: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        src_mask: torch.BoolTensor,
        tgt_mask: Optional[torch.BoolTensor] = None
    ):
        """
        The final hidden state for the last token index contains the next token predictive distribution
        """
        # (n_batches, sequence_length, hidden_dim)
        # TODO this multiplication seems to result in the first or last output logit to be the first or last voca entry
        x = self.embed(input_tokens) * math.sqrt(self.hidden_dim)
        x = self.positional_encoding(x)
        x = self.dropout(x)

        for decoder_block in self.decoder_blocks:
            x = decoder_block(x, encoder_hidden_states, src_mask, tgt_mask)

        # (n_batches, sequence_length, vocab_size)
        logits = self.output_layer(x)
        output = self.softmax(logits)
        return output


class TransformerDecoderBlock(nn.Module):
    def __init__(self, hidden_dim: int, ff_dim: int, num_heads: int, dropout_p: float):
        super().__init__()

        self.cross_mha = MultiHeadAttention(hidden_dim, num_heads)
        self.self_mha = MultiHeadAttention(hidden_dim, num_heads)
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_dim, ff_dim), nn.ReLU(), nn.Linear(ff_dim, hidden_dim),
        )

        self.dropout1 = nn.Dropout(p=dropout_p)
        self.dropout2 = nn.Dropout(p=dropout_p)
        self.dropout3 = nn.Dropout(p=dropout_p)

        self.layer_norm1 = nn.LayerNorm(hidden_dim)
        self.layer_norm2 = nn.LayerNorm(hidden_dim)
        self.layer_norm3 = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        x: torch.Tensor,
        encoder_hidden_states: torch.FloatTensor,
        src_mask: torch.BoolTensor,
        tgt_mask: Optional[torch.BoolTensor] = None,
    ):
        """
        x is of shape(n_batches, decoded_sequence_length, hidden_dim)
        encoder_hidden_states is of shape(n_batches, src_sequence_length, hidden_dim)
        """
        # Self attention (with future masking during training)
        output = self.dropout1(self.self_mha.forward(x, mask=tgt_mask))
        x = self.layer_norm1(x + output)

        # Cross or encoder-decoder attention
        output = self.dropout2(
            self.cross_mha.forward(
                x, encoder_hidden_states=encoder_hidden_states, mask=src_mask,
            )
        )
        x = self.layer_norm2(x + output)

        # Feed forward layers
        output = self.dropout3(self.feed_forward(x))
        x = self.layer_norm3(x + output)
        return x


class TestTransformerDecoder(unittest.TestCase):
    def test_transformer_decoder_inference(self):
        """ Test two forward passes, simulating two inference decoding steps"""
        seed = 1
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)

        with torch.no_grad():
            n_batches, seq_len, hidden_dim, vocab_size = 2, 10, 512, 2000

            # Prepare fake encoder hidden states and attention masks
            encoder_hidden_states = torch.randn((n_batches, seq_len, hidden_dim))
            src_mask = torch.BoolTensor(
                [[1, 1, 1, 1, 1, 1, 1, 1, 0, 0], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]
            )

            # Initialize decoder input with a bos token (0) for each sequence in the batch
            decoder_input_ids = torch.empty(n_batches, 1, dtype=torch.int).fill_(0)
            decoder = TransformerDecoder(
                embedding=torch.nn.Embedding(vocab_size, hidden_dim),
                hidden_dim=hidden_dim,
                ff_dim=2048,
                num_heads=8,
                num_layers=2,
                dropout_p=0.1,
                vocab_size=vocab_size,
            )
            decoder._reset_parameters()
            decoder.eval()

            # Perform first forward pass
            output = decoder.forward(
                decoder_input_ids, encoder_hidden_states, src_mask, tgt_mask=None
            )
            self.assertEqual(output.shape, (n_batches, 1, vocab_size))
            self.assertEqual(torch.any(output == 1), False)

            # Append argmax prediction to the decoder input
            predicted_token_ids = torch.argmax(output, dim=-1)
            decoder_input_ids = torch.cat((decoder_input_ids, predicted_token_ids), dim=-1)

            # Perform second forward pass
            output = decoder.forward(
                decoder_input_ids, encoder_hidden_states, src_mask, tgt_mask=None
            )
            self.assertEqual(output.shape, (n_batches, 2, vocab_size))
            decoder_input_ids = torch.cat((decoder_input_ids, predicted_token_ids), dim=-1)
            # TODO this is weird, I expect random token indices here
            raise Exception("Decoder softmax output should not always be 0")
            torch.testing.assert_allclose(
                decoder_input_ids,
                torch.Tensor([
                    [0, 0, 0],
                    [0, 0, 0]
                ])
            )


if __name__ == "__main__":
    unittest.main()
