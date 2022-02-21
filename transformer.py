import unittest
from typing import Optional

import torch
from torch import nn

from vocabulary import Vocabulary
from multi_head_attention import MultiHeadAttention
from positional_encodings import SinusoidEncoding
from encoder import TransformerEncoder
from decoder import TransformerDecoder


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
        dropout_p,
    ):
        super().__init__()
        # Share encoder and decoder embeddings weights
        self.embed = nn.Embedding(vocab_size, hidden_dim, padding_idx=padding_idx)
        self.encoder = TransformerEncoder(self.embed, hidden_dim, ff_dim, num_heads, num_layers, dropout_p)
        self.decoder = TransformerDecoder(self.embed, hidden_dim, ff_dim, num_heads, num_layers, vocab_size, dropout_p)

        self.padding_idx = padding_idx
        self.bos_idx = bos_idx
        self.max_decoding_length = max_decoding_length

    def forward(
            self,
            input_ids: torch.Tensor,
            target_ids: Optional[torch.Tensor] = None,
            src_mask: Optional[torch.BoolTensor]= None
    ):
        encoder_output = self.encoder.forward(input_ids)  # (n_batches, seq_len, hidden_dim)

        # Initialize decoder input with a bos token for each to-be-generated sequence in the batch
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
        batch = [
            "Hello my name is Joris and I was born with the name Joris.",
            "Another, shorter sequence in the batch.",
        ]
        en_vocab = Vocabulary(batch)
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
            dropout_p=0.1,
        )
        input_batch = torch.IntTensor(en_vocab.batch_encode(batch))
        src_mask = input_batch == en_vocab.token2index[en_vocab.PAD]
        output = transformer.forward(input_batch, src_mask=src_mask)
        self.assertEqual(output.shape, (1, 16, 512))


if __name__ == "__main__":
    unittest.main()
