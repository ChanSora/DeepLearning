import torch
import torch.nn as nn
import torch.functional as F
import random

from nanogpt_2_model import *

url = '/Users/chansora/fun/wechat_data/Les.txt'

words = list(open(url, 'r').read())
charset = sorted(set(words))
vocab_size = len(charset)
print(vocab_size)

stoi = {char:idx for idx, char in enumerate(charset)}
itos = {idx:char for idx, char in enumerate(charset)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[c] for c in l])

torch.manual_seed(random.randint(1, 2147483647))

def main():
    model = torch.load('nanogpt_v1.pth', weights_only=False)
    print(decode(model.generate(torch.zeros((1, 1), dtype=int, device=device), max_tokens=5000)[0].tolist()))

    
if __name__ == '__main__':
    main()