[![Transformer unit tests](https://github.com/jsbaan/transformer-from-scratch/actions/workflows/unit-tests.yml/badge.svg)](https://github.com/jsbaan/transformer-from-scratch/actions/workflows/unit-tests.yml)
[![Mypy Type Checking](https://github.com/jsbaan/transformer-from-scratch/actions/workflows/mypy-type-checking.yml/badge.svg)](https://github.com/jsbaan/transformer-from-scratch/actions/workflows/mypy-type-checking.yml)
<a href="https://github.com/psf/black"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>

# Implementing A Transformer From Scratch
To get intimately familiar with the nuts and bolts of transformers I decided to implement the original architecture from [Attention Is All You Need](https://arxiv.org/abs/1706.03762).

This repo accompanies the blogpost [Implementing a Transformer From Scratch: 7 Things You Didn't Know About the Transformer](https://jorisbaan.medium.com/7-things-you-didnt-know-about-the-transformer-a70d93ced6b2). I wrote this blogpost to highlight things that I learned in the process and that I found particularly surprising or insightful.

# Features:
This repo contains the following features: 
- The simplest imaginable vocabulary ([vocabulary.py](https://github.com/jsbaan/transformer-from-scratch/blob/main/vocabulary.py)).
- The simplest imaginable (batch) tokenizer ([vocabulary.py](https://github.com/jsbaan/transformer-from-scratch/blob/main/vocabulary.py)).
- TransformerEncoder and EncoderBlock classes ([encoder.py](https://github.com/jsbaan/transformer-from-scratch/blob/main/encoder.py)).
- TransformerDecoder and DecoderBlock classes ([decoder.py](https://github.com/jsbaan/transformer-from-scratch/blob/main/decoder.py)).
- Transformer main class ([transformer.py](https://github.com/jsbaan/transformer-from-scratch/blob/main/transformer.py)).
- Train script with a unit test that (over)fits a synthetic copy dataset ([train.py](https://github.com/jsbaan/transformer-from-scratch/blob/main/train.py)).
- MultiHeadAttention class with scaled dot product and masking ([multi_head_attention.py](https://github.com/jsbaan/transformer-from-scratch/blob/main/multi_head_attention.py)).
- SinusoidEncoding class for positional encodings ([positional_encodings.py](https://github.com/jsbaan/transformer-from-scratch/blob/main/positional_encoding.py)).
- Utility functions to construct masks and batches ([utils.py](https://github.com/jsbaan/transformer-from-scratch/blob/main/utils.py)).
- Learning rate scheduler ([lr_scheduler.py](https://github.com/jsbaan/transformer-from-scratch/blob/main/lr_scheduler.py)).
- Basic unit tests for each class. Running a file (e.g. `python encoder.py`) will execute its unit tests.
- Type checking using [mypy](https://mypy.readthedocs.io/en/stable/). 
- Code formatting using [black](https://github.com/psf/black).
- Automatic execution of unit tests and type checks using Github Actions.
- No dependencies except Python 3.9 and PyTorch 1.9.1 (though basically any version should work).