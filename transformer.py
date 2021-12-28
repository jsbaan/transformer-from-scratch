import unittest
from typing import Optional

import torch
from torch import nn

from vocabulary import Vocabulary
from multi_head_attention import MultiHeadAttention
from positional_encodings import SinusoidEncoding


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


class TransformerDecoder(nn.Module):
    def __init__(
        self,
        embedding: torch.nn.Embedding,
        vocab_size: int,
        hidden_dim: int,
        ff_dim: int,
        num_heads: int,
        num_layers: int,
    ):
        super().__init__()
        self.embed = embedding
        self.positional_encoding = SinusoidEncoding(hidden_dim)
        self.decoder_blocks = nn.ModuleList(
            [TransformerDecoderBlock(hidden_dim, ff_dim, num_heads) for _ in range(num_layers)]
        )
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
        # TODO add masked self-attention here
        self.cross_mha = MultiHeadAttention(hidden_dim, num_heads)
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_dim, ff_dim), nn.ReLU(), nn.Linear(ff_dim, hidden_dim)
        )
        # TODO verify this, I don't fully understand layer norm
        self.layer_norm = nn.LayerNorm(hidden_dim)
        # TODO add cross-attention here

    def forward(self, x: torch.Tensor, encoder_hidden_states: torch.Tensor):
        # TODO add masked self attention here
        mha_output = self.cross_mhamulti_head_attention.forward(x)  # (n_batches, sequence_length, hidden_dim)
        x = self.layer_norm(x + mha_output)

        ff_output = self.feed_forward(x)  # (n_batches, sequence_length, hidden_dim)
        x = self.layer_norm(x + ff_output)

        # TODO add cross-attention here
        return x


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
        self.decoder = TransformerDecoder(self.embed, vocab_size, hidden_dim, ff_dim, num_heads, num_layers)

        # Tie embedding weights to decoder output layer weights
        self.decoder.output_layer.weight = self.embed.weight.T

        self.padding_idx = padding_idx
        self.bos_idx = bos_idx
        self.max_decoding_length = max_decoding_length

    def forward(self, input_ids: torch.Tensor, target_ids: Optional[torch.Tensor] = None):
        encoder_output = self.encoder.forward(input_ids)

        # Prepare decoder input batch containing a beginning of sequence token per to-be-generated sequence
        n_batches, _ = input_ids.shape
        decoder_input_ids = torch.empty(n_batches, 1).fill_(self.bos_idx)

        decoded_sequences = self.decoder.forward(decoder_input_ids, encoder_output)
        for i in range(self.max_decoding_length):
            decoded_sequences = self.decoder.forward(decoded_sequences, encoder_output)

            # TODO add bookkeeping to check whether all sequences have generated an EOS token
        return decoder_input_ids


class TestTransformer(unittest.TestCase):
    def test_transformer_encoder(self):
        # Create vocabulary and special token indices given a dummy corpus
        corpus = ["Hello my name is Joris and I was born with the name Joris."]
        en_vocab = Vocabulary(corpus)
        en_vocab_size = len(en_vocab.token2index.items())
        padding_idx = en_vocab.token2index[en_vocab.PAD]
        bos_idx = en_vocab.token2index[en_vocab.BOS]

        # Initialize a transformer encoder (qkv_dim is automatically set to hidden_dim // num_heads)
        encoder = TransformerEncoder(
            embedding=torch.nn.Embedding(en_vocab_size, 512),
            hidden_dim=512,
            ff_dim=2048,
            num_heads=8,
            num_layers=2,
        )
        input_batch = torch.IntTensor([en_vocab.encode(corpus[0])])
        output = encoder.forward(input_batch)
        self.assertEqual(output.shape, (1, 16, 512))

        # TODO test multi-sequence batch with padding

    def test_transformer(self):
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
        encoder = Transformer(
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
        output = encoder.forward(input_batch)
        self.assertEqual(output.shape, (1, 16, 512))


if __name__ == "__main__":
    unittest.main()
