import re
import unittest

from typing import List, Optional
from torch import IntTensor


class Vocabulary:
    BOS = "BOS"
    EOS = "EOS"
    PAD = "PAD"

    def __init__(self, list_of_sentences: Optional[List[str]]):
        self.token2index = {self.BOS: 0, self.EOS: 1, self.PAD: 2}
        self.index2token = {v: k for k, v in self.token2index.items()}
        for sentence in list_of_sentences:
            self.add_tokens(self.tokenize(sentence))

    def add_tokens(self, tokens: List[str]):
        """ Adds tokens the vocabulary """
        for token in tokens:
            if token not in self.token2index:
                i = len(self.token2index.items())
                self.token2index[token] = i
                self.index2token[i] = token

    def tokenize(self, sentence: str, add_special_tokens: bool = True):
        """ Split on all tokens and punctuation and adds BOS and EOS token """
        tokens = re.findall(r"\w+|[^\s\w]+", sentence)
        if add_special_tokens:
            tokens = [self.BOS] + tokens + [self.EOS]
        return tokens

    def encode(self, sentence: str, add_special_tokens: bool = True):
        """ Converts a sentence to a list of token indices in the vocabulary """
        tokens = self.tokenize(sentence, add_special_tokens)
        return [self.token2index[token] for token in tokens]

    def batch_encode(self, sentences: List[str], padding=True, add_special_tokens: bool = False):
        """ Convert a list of string sentences to nested list of token indices. Optionally padding & bos+eos tokens """
        tokenized_sentences = [self.encode(sentence, add_special_tokens) for sentence in sentences]
        if padding:
            max_length = max([len(tokens) for tokens in tokenized_sentences])
            tokenized_sentences = [s + ((max_length-len(s)) * [self.token2index[self.PAD]]) for s in tokenized_sentences]
        return tokenized_sentences


class TestVocabulary(unittest.TestCase):
    maxDiff = None

    def test_tokenize(self):
        input_sequence = "Hello my name is Joris and I was born with the name Joris."
        output = Vocabulary([]).tokenize(input_sequence)
        self.assertEqual(
            [
                "BOS",
                "Hello",
                "my",
                "name",
                "is",
                "Joris",
                "and",
                "I",
                "was",
                "born",
                "with",
                "the",
                "name",
                "Joris",
                ".",
                "EOS",
            ],
            output,
        )

    def test_init_vocab(self):
        input_sentences = ["Hello my name is Joris and I was born with the name Joris."]
        vocab = Vocabulary(input_sentences)
        expected = {
            "BOS": 0,
            "EOS": 1,
            "PAD": 2,
            "Hello": 3,
            "my": 4,
            "name": 5,
            "is": 6,
            "Joris": 7,
            "and": 8,
            "I": 9,
            "was": 10,
            "born": 11,
            "with": 12,
            "the": 13,
            ".": 14,
        }
        self.assertEqual(vocab.token2index, expected)

    def test_encode(self):
        input_sentences = ["Hello my name is Joris and I was born with the name Joris."]
        vocab = Vocabulary(input_sentences)
        output = vocab.encode(input_sentences[0])
        self.assertEqual(output, [0, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 5, 7, 14, 1])

    def test_encode_no_special_tokens(self):
        input_sentences = ["Hello my name is Joris and I was born with the name Joris."]
        vocab = Vocabulary(input_sentences)
        output = vocab.encode(input_sentences[0], add_special_tokens=False)
        self.assertEqual(output, [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 5, 7, 14])

    def test_batch_encode(self):
        input_sentences = ["This is one sentence", "This is another, much longer sentence", "Short sentence"]
        vocab = Vocabulary(input_sentences)
        output = vocab.batch_encode(input_sentences, add_special_tokens=False)
        self.assertEqual(output, [[3, 4, 5, 6, 2, 2, 2], [3, 4, 7, 8, 9, 10, 6], [11, 6, 2, 2, 2, 2, 2]])


if __name__ == "__main__":
    unittest.main()
