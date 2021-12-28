import math
import unittest
from typing import Optional

import torch


class SinusoidEncoding:
    def __init__(self, n_batches: int, sequence_length: int, embedding_dim: int):
        self.n_batches = n_batches
        self.embedding_dim = embedding_dim
        self.sequence_length = sequence_length

    def construct(
        self,
        n_batches: Optional[int] = None,
        sequence_length: Optional[int] = None,
        embedding_dim: Optional[int] = None,
    ):
        """ Relative positional encodings using sinusoids """
        n_batches = n_batches if n_batches else self.n_batches
        sequence_length = sequence_length if sequence_length else self.sequence_length
        embedding_dim = embedding_dim if embedding_dim else self.embedding_dim

        positional_encodings = torch.zeros(sequence_length, embedding_dim)
        for token_idx in range(sequence_length):
            for dim_idx in range(embedding_dim):
                positional_encodings[token_idx, dim_idx] = (
                    math.sin(token_idx / pow(10.000, (2 * dim_idx) / embedding_dim))
                    if dim_idx % 2 == 0
                    else math.cos(token_idx / pow(10.000, (2 * dim_idx) / embedding_dim))
                )
        return positional_encodings.unsqueeze(0).repeat(n_batches, 1, 1)


class TestSinusoidEncoding(unittest.TestCase):
    def test_create_embedding(self):
        encoding = SinusoidEncoding(1, 2, 3).construct()
        expected = torch.Tensor([[[0.0, 1.0, 0.0], [0.8415, 0.9769, 0.0464]]])
        torch.testing.assert_close(encoding, expected, rtol=10e-5, atol=10e-5)

    def test_create_embedding_more_batches(self):
        encoding = SinusoidEncoding(2, 2, 3).construct()
        expected = torch.Tensor([[[0.0, 1.0, 0.0], [0.8415, 0.9769, 0.0464]],
                                 [[0.0, 1.0, 0.0], [0.8415, 0.9769, 0.0464]]])
        torch.testing.assert_close(encoding, expected, rtol=10e-5, atol=10e-5)


if __name__ == "__main__":
    unittest.main()
