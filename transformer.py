import unittest
from typing import Optional

import torch
from torch import nn

from vocabulary import Vocabulary
from multi_head_attention import MultiHeadAttention
from positional_encodings import SinusoidEncoding


class Transformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        padding_idx: int,
        bos_idx: int,
        hidden_dim: int,
        ff_dim: int,
        num_heads: int,
        num_layers: int,
        max_decoding_length: int,
    ):
        super().__init__()
        # Share encoder and decoder embeddings weights
        self.embed = nn.Embedding(vocab_size, hidden_dim, padding_idx=padding_idx)
        self.encoder = TransformerEncoder(self.embed, hidden_dim, ff_dim, num_heads, num_layers)
        self.decoder = TransformerDecoder(self.embed, hidden_dim, ff_dim, num_heads, num_layers, vocab_size)

        # Tie embedding weights to decoder output layer weights
        self.decoder.output_layer.weight = self.embed.weight.T

        self.padding_idx = padding_idx
        self.bos_idx = bos_idx
        self.max_decoding_length = max_decoding_length

    def forward(self, input_ids: torch.Tensor, target_ids: Optional[torch.Tensor] = None):
        encoder_output = self.encoder.forward(input_ids)  # (n_batches,

        # Prepare decoder input batch containing a beginning of sequence token per to-be-generated sequence
        n_batches, _ = input_ids.shape
        decoder_input_ids = torch.empty(n_batches, 1).fill_(self.bos_idx)

        decoded_sequences = self.decoder.forward(decoder_input_ids, encoder_output)
        for i in range(self.max_decoding_length):
            decoded_sequences = self.decoder.forward(decoded_sequences, encoder_output)

            # TODO add bookkeeping to check whether all sequences have generated an EOS token
        return decoder_input_ids


class TestTransformer(unittest.TestCase):

    def test_transformer(self):
        # TODO finish/rewrite
        # Create vocabulary and special token indices given a dummy corpus
        corpus = [
            "Hello my name is Joris and I was born with the name Joris.",
            "Hallo mijn naam is Joris en ik ben geboren met de naam Joris",
        ]
        en_vocab = Vocabulary(corpus)
        en_vocab_size = len(en_vocab.token2index.items())
        padding_idx = en_vocab.token2index[en_vocab.PAD]
        bos_idx = en_vocab.token2index[en_vocab.BOS]

        # Initialize a transformer encoder (qkv_dim is automatically set to hidden_dim // num_heads)
        transformer = Transformer(
            vocab_size=en_vocab_size,
            padding_idx=padding_idx,
            bos_idx=bos_idx,
            hidden_dim=512,
            ff_dim=2048,
            num_heads=8,
            num_layers=2,
            max_decoding_length=50,
        )
        input_batch = torch.IntTensor([en_vocab.encode(corpus[0])])
        output = transformer.forward(input_batch)
        self.assertEqual(output.shape, (1, 16, 512))


if __name__ == "__main__":
    unittest.main()
