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
from utils import construct_future_mask


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
        input_tokens: torch.IntTensor,
        encoder_hidden_states: torch.Tensor,
        src_padding_mask: Optional[torch.BoolTensor] = None,
        future_mask: Optional[torch.BoolTensor] = None,
    ):
        """
        The final hidden state for the last token index contains the next token predictive distribution
        """
        # (batch_size, sequence_length, hidden_dim)
        x = self.embed(input_tokens) * math.sqrt(self.hidden_dim)
        x = self.positional_encoding(x)
        x = self.dropout(x)

        for decoder_block in self.decoder_blocks:
            x = decoder_block(x, encoder_hidden_states, src_padding_mask, future_mask)

        # (batch_size, sequence_length, vocab_size)
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
        src_padding_mask: Optional[torch.BoolTensor] = None,
        future_mask: Optional[torch.BoolTensor] = None,
    ):
        """
        x is of shape(batch_size, decoded_sequence_length, hidden_dim)
        encoder_hidden_states is of shape(batch_size, src_sequence_length, hidden_dim)
        """
        # Self attention (with future masking during training)

        output = self.dropout1(self.self_mha.forward(x, future_mask=future_mask))
        x = self.layer_norm1(x + output)

        # Cross or encoder-decoder attention
        output = self.dropout2(
            self.cross_mha.forward(
                x,
                encoder_hidden_states=encoder_hidden_states,
                src_padding_mask=src_padding_mask,
            )
        )
        x = self.layer_norm2(x + output)

        # Feed forward layers
        output = self.dropout3(self.feed_forward(x))
        x = self.layer_norm3(x + output)
        return x


class TestTransformerDecoder(unittest.TestCase):
    def test_transformer_decoder_inference(self):
        """
        Test two forward passes, simulating two inference decoding steps
        """
        seed = 0
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)

        with torch.no_grad():
            batch_size, src_seq_len, hidden_dim, vocab_size = 2, 10, 512, 2000

            # Prepare fake encoder hidden states and padding masks
            # TODO is it required to scale encoder hidden states with this factor?
            encoder_output = torch.randn(
                (batch_size, src_seq_len, hidden_dim)
            ) * math.sqrt(hidden_dim)
            src_padding_mask = torch.BoolTensor(
                [[1, 1, 1, 1, 1, 1, 1, 1, 0, 0], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]
            )

            # Initialize the decoder, perform xavier init and set to evaluation mode
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

            # Prepare decoder input, mask, perform a decoding step, take the argmax over the softmax of the last token
            # and iteratively feed the input+prediction back in.
            decoder_input = torch.IntTensor([[0], [0]])
            future_mask = None
            for i in range(3):
                decoder_output = decoder(
                    decoder_input,
                    encoder_output,
                    src_padding_mask=src_padding_mask,
                    future_mask=future_mask,
                )
                predicted_tokens = torch.argmax(
                    decoder_output[:, -1, :], dim=-1
                ).unsqueeze(1)
                decoder_input = torch.cat((decoder_input, predicted_tokens), dim=-1)
                future_mask = construct_future_mask(decoder_input.shape[1])

                self.assertEqual(decoder_output.shape, (batch_size, i + 1, vocab_size))
                # softmax entropy should not be 0
                self.assertEqual(torch.any(decoder_output == 1), False)
                # token predictions should not all be 0
                # TODO this is weird, I expect random token indices here
                self.assertEqual(torch.all(decoder_input == 0), False)


if __name__ == "__main__":
    unittest.main()
