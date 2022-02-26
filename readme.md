[![Transformer unit tests](https://github.com/jsbaan/transformer-from-scratch/actions/workflows/unit-tests.yml/badge.svg)](https://github.com/jsbaan/transformer-from-scratch/actions/workflows/unit-tests.yml)
<a href="https://github.com/psf/black"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>

# Implementing A Transformer From Scratch
To get intimately familiar with the nuts and bolts of transformers, I implemented the original architecture as proposed in [Attention Is All You Need](https://arxiv.org/abs/1706.03762). No dependencies except [PyTorch](https://pytorch.org/get-started/locally/). This repo comes with a blogpost that I wrote about the implementation details that surprised me, and thought were worth highlighting. You can find the blogpost **here**.

WORK IN PROGRESS!

# Features:
- The simplest imaginable vocabulary (vocabulary.py)
- The simplest imaginable (batch) tokenizer (vocabulary.py)
- TransformerEncoder and EncoderBlock classes (encoder.py)
- TransformerDecoder and DecoderBlock classes (decoder.py)
- Transformer main class (transformer.py)
- MultiHeadAttention class with scaled dot product and masking (multi_head_attention.py)
- SinusoidEncoding class for positional encoding (positional_encoding.py)
- Basic unit tests for each class
- Type checking
- Code formatted using [black](https://github.com/psf/black)
- No dependencies except Python 3.9 and PyTorch 1.9.1 (though any version should work).

# To do:
### Code
- Write minimal training loop to test if model can fit a tiny "copy" dataset
  - Optimizer, learning rate scheduler, loss, etc.
  - Check whether BOS and EOS tokens should be added to each training sentence; remove this from the tokenization if not.

### Writing
- Add details about masking in MHA
- Add details about the formal meaning of the word projection
- Expand on layer normalization; most people know it normalizes the batch somehow but what does that actually do?
- Write about relation between transformer and popular models such as bert or gpt
- Think about title, motivation and framing

### Bonus to do
- Replace examples in the vocabulary unit test.
- (Multi-)GPU support