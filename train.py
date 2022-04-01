from typing import OrderedDict
import torch
import torch.nn as nn
from torchtext.datasets import WikiText2
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

import matplotlib.pyplot as plt
from model import Transformer
from tqdm import tqdm
import math


def data_process(raw_text_iter, vocab, tokenizer):
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


def dataloader(data, batch_size, bptt, device):
    data_batch = batchify(data, batch_size)
    loader = list(get_batch(data_batch, i, bptt) for i in range(0, data_batch.size(0) - 1, bptt))
    loader = list((x.to(device), y.to(device)) for x, y in loader)
    return loader


def train(model, n_tokens, train_loader, val_loader=None):
    criterion = nn.CrossEntropyLoss()
    lr = 4  # learning rate
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)
    n_epoch = 1

    total_loss = 0
    cnt = 0
    loss_h = []
    ppl_h = []

    save_ppl = False
    saved = True
    if saved:
        model.load_state_dict(torch.load("./saved.bin"))
    else:
        for epoch in range(n_epoch):
            for d, t in tqdm(train_loader):
                out = model(d, 0)
                loss = criterion(out.view(-1, n_tokens), t)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                optimizer.step()
                # scheduler.step()

                total_loss += loss.item()
                cnt += 1
                if cnt == 10:
                    loss_h.append(total_loss / 10)
                    cnt = 0
                    total_loss = 0

                    if save_ppl:
                        ppl = 0
                        val_cnt = 0
                        for vd, vt in tqdm(val_loader):
                            out = model(vd)
                            ppl += vd.size(0) * vd.size(1) * criterion(out.view(-1, n_tokens), vt).item()
                            val_cnt += vd.size(0) * vd.size(1)
                        ppl_h.append(math.exp(ppl / val_cnt))

        plt.plot(loss_h)
        plt.title("Train Loss")
        plt.savefig("Loss.png")
        plt.cla()

        if save_ppl:
            plt.plot(ppl_h)
            plt.savefig("ppl.png")

        torch.save(model.state_dict(), "./saved.bin")


def save_quantized(model):
    int_state = OrderedDict()
    with torch.no_grad():
        for x, y in model.state_dict().items():
            if "weight" in x or "bias" in x:
                int_state[x] = y.char()

    torch.save(int_state, "./q_saved.bin")


def eval(model, n_tokens, data_loader):
    ppl = 0
    total_time = 0
    eval_cnt = 0
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for d, t in tqdm(data_loader):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            out = model(d, 0)
            end.record()
            torch.cuda.synchronize()
            total_time += start.elapsed_time(end)
            ppl += d.size(0) * d.size(1) * criterion(out.view(-1, n_tokens), t).item()
            eval_cnt += d.size(0) * d.size(1)
    print()
    print("PPL:")
    print(math.exp(ppl / eval_cnt))
    print("Total Time (ms):")
    print(total_time)
    print()


def apply_quantization(model, n_tokens, d_model, n_head, n_layers, device):
    model_2 = Transformer(n_tokens, d_model, n_head, n_layers, True)
    model_2.load_state_dict(model.state_dict())
    for i, layer in enumerate(model.encoder_layers):
        model_2.encoder_layers[i].Attention.a = layer.Attention.a
        model_2.encoder_layers[i].Attention.b = layer.Attention.b
        model_2.encoder_layers[i].FeedForward.a = layer.FeedForward.a
        model_2.encoder_layers[i].FeedForward.b = layer.FeedForward.b

    model_2.to(device)
    return model_2


def main():
    train_iter = WikiText2(split='train')
    tokenizer = get_tokenizer('basic_english')
    vocab = build_vocab_from_iterator(map(tokenizer, train_iter), specials=['<unk>'])
    vocab.set_default_index(vocab['<unk>'])

    train_iter, val_iter, test_iter = WikiText2()
    train_data = data_process(train_iter, vocab, tokenizer)
    val_data = data_process(val_iter, vocab, tokenizer)
    test_data = data_process(test_iter, vocab, tokenizer)

    n_tokens, d_model, n_head, n_layers = len(vocab), 256, 2, 2
    batch_size, bptt = 64, 32
    device = torch.device("cuda:2")

    model = Transformer(n_tokens, d_model, n_head, n_layers, False)
    model.to(device)

    train_loader = dataloader(train_data, batch_size, bptt, device)
    train(model, n_tokens, train_loader)

    del train_loader

    test_loader = dataloader(test_data, batch_size, bptt, device)
    eval(model, n_tokens, test_loader)

    model = apply_quantization(model, n_tokens, d_model, n_head, n_layers, device)
    eval(model, n_tokens, test_loader)
    return


if __name__ == "__main__":
    main()
