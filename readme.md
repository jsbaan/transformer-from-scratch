[![Transformer unit tests](https://github.com/jsbaan/transformer-from-scratch/actions/workflows/unit-tests.yml/badge.svg)](https://github.com/jsbaan/transformer-from-scratch/actions/workflows/unit-tests.yml)

# Implementing A Transformer From Scratch
To get intimately familiar with the nuts and bolts of transformers, I implemented a bare-bone version of the original transformer 
proposed in [Attention Is All You Need](https://arxiv.org/abs/1706.03762). No dependencies except [PyTorch](https://pytorch.org/get-started/locally/) and not intended for real-world use.

WORK IN PROGRESS!

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
- No dependencies except Python 3.9 and PyTorch 1.9.1 (though any version should work). See requirements.txt.

### Bonus to do
- Replace examples in the vocabulary unit test.
- (Multi-)GPU support
- Run unit tests on each commit using Github Actions 
  - Refactor unit tests to separate folder and files
  - Run pylint on codebase