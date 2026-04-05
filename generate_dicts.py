import torch
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from datasets import load_dataset
import os

print("Loading dataset...")
dataset = load_dataset('cfilt/iitb-english-hindi')
en_tokenizer = get_tokenizer('basic_english')
hi_tokenizer = lambda x: x.split()

def yield_tokens(data_iter, lang):
    for example in data_iter:
        yield en_tokenizer(example['translation']['en']) if lang == 'en' else hi_tokenizer(example['translation']['hi'])

print("Building vocabs...")
limit = 10000
en_vocab = build_vocab_from_iterator(yield_tokens(dataset['train'].select(range(limit)), 'en'), specials=['<unk>', '<pad>', '<sos>', '<eos>'])
hi_vocab = build_vocab_from_iterator(yield_tokens(dataset['train'].select(range(limit)), 'hi'), specials=['<unk>', '<pad>', '<sos>', '<eos>'])

print("Saving plain dictionaries (no-torchtext dependency)...")
torch.save({'stoi': dict(en_vocab.get_stoi()), 'itos': en_vocab.get_itos()}, 'en_vocab_dict.pt')
torch.save({'stoi': dict(hi_vocab.get_stoi()), 'itos': hi_vocab.get_itos()}, 'hi_vocab_dict.pt')
print("Vocab Dictionaries generated successfully.")
