from datasets import load_dataset
import torch
import re
from collections import Counter

def basic_tokenizer(text):
    text = text.lower()
    text = re.sub(r"([.,!?\"':;])", r" \1 ", text)
    return text.split()

def hindi_tokenizer(text):
    return text.split()

print("Loading dataset for portable vocab...")
dataset = load_dataset('cfilt/iitb-english-hindi')
limit = 10000

en_counter = Counter()
hi_counter = Counter()

print(f"Building counters for first {limit} examples...")
for i in range(limit):
    example = dataset['train'][i]['translation']
    en_counter.update(basic_tokenizer(example['en']))
    hi_counter.update(hindi_tokenizer(example['hi']))

def build_dict(counter, specials=['<unk>', '<pad>', '<sos>', '<eos>']):
    stoi = {word: i + len(specials) for i, (word, _) in enumerate(counter.items())}
    for i, spec in enumerate(specials):
        stoi[spec] = i
    itos = {i: word for word, i in stoi.items()}
    return stoi, itos

print("Constructing dictionaries...")
en_stoi, en_itos = build_dict(en_counter)
hi_stoi, hi_itos = build_dict(hi_counter)

print("Saving portable dictionaries...")
torch.save({'stoi': en_stoi, 'itos': en_itos}, 'en_vocab_portable.pt')
torch.save({'stoi': hi_stoi, 'itos': hi_itos}, 'hi_vocab_portable.pt')
print("Portable Vocabs saved. No torchtext dependency required.")
