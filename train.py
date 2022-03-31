import torch
import torch.nn as nn
from torchtext.datasets import WikiText2
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

import matplotlib.pyplot as plt
from model import Transformer
from tqdm import tqdm
import math

train_iter = WikiText2(split='train')
tokenizer = get_tokenizer('basic_english')
vocab = build_vocab_from_iterator(map(tokenizer, train_iter), specials=['<unk>'])
vocab.set_default_index(vocab['<unk>'])


def data_process(raw_text_iter):
    """Converts raw text into a flat Tensor."""
    data = [torch.tensor(vocab(tokenizer(item)), dtype=torch.long) for item in raw_text_iter]
    return torch.cat(tuple(filter(lambda t: t.numel() > 0, data)))


def generate_square_subsequent_mask(sz: int):
    """Generates an upper-triangular matrix of -inf, with zeros on diag."""
    return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)


def batchify(data, bsz):
    seq_len = data.size(0) // bsz
    data = data[:seq_len * bsz]
    data = data.view(bsz, seq_len).t().contiguous()
    return data


def get_batch(source, i, bptt):
    seq_len = min(bptt, len(source) - 1 - i)
    data = source[i:i+seq_len]
    target = source[i+1:i+1+seq_len].reshape(-1)
    return data, target


train_iter, val_iter, test_iter = WikiText2()
train_data = data_process(train_iter)
val_data = data_process(val_iter)
test_data = data_process(test_iter)

device = torch.device("cuda:2")
batch_size = 64
eval_batch_size = 64
train_data = batchify(train_data, batch_size).to(device)  # shape [seq_len, batch_size]
val_data = batchify(val_data, eval_batch_size).to(device)
test_data = batchify(test_data, eval_batch_size).to(device)

n_tokens, d_model, n_head, n_layers = len(vocab), 256, 2, 2

model = Transformer(n_tokens, d_model, n_head, n_layers, False)
model.to(device)
criterion = nn.CrossEntropyLoss()
lr = 5  # learning rate
optimizer = torch.optim.SGD(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)
bptt = 30
n_epoch = 2

total_loss = 0
cnt = 0
loss_h = []
ppl_h = []

save_ppl = False
saved = False
if saved:
    model.load_state_dict(torch.load("./saved.bin"))
else:
    for epoch in range(n_epoch):
        for i in tqdm(range(0, train_data.size(0) - 1, bptt)):
            d, t = get_batch(train_data, i, bptt)
            out = model(d, generate_square_subsequent_mask(batch_size).to(device))
            loss = criterion(out.view(-1, n_tokens), t)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()

            total_loss += loss.item()
            cnt += 1
            if cnt == 100:
                loss_h.append(total_loss / 100)
                cnt = 0
                total_loss = 0

                if save_ppl:
                    ppl = 0
                    for j in range(0, val_data.size(0) - 1, bptt):
                        vd, vt = get_batch(val_data.to(device), j, bptt)
                        out = model(vd)
                        ppl += vd.size(0) * criterion(out.view(-1, n_tokens), vt).item()
                    ppl_h.append(math.exp(ppl / (val_data.size(0) - 1)))

    plt.plot(loss_h)
    plt.savefig("Loss.png")
    plt.cla()

    if save_ppl:
        plt.plot(ppl_h)
        plt.savefig("ppl.png")

    torch.save(model.state_dict(), "./saved.bin")

ppl = 0
for j in range(0, test_data.size(0) - 1, bptt):
    vd, vt = get_batch(test_data, j, bptt)
    out = model(vd, generate_square_subsequent_mask(eval_batch_size).to(device))
    ppl += vd.size(0) * criterion(out.view(-1, n_tokens), vt).item()
print(math.exp(ppl / (test_data.size(0) - 1)))
