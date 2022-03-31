import torch
from time import time

SIZE = 500
n_test = 100


def mm_int8(X, Y):
    X = X.char()
    Y = Y.char()
    a = time()
    Z = torch.mm(X, Y)
    return time() - a


def mm_int16(X, Y):
    X = X.long()
    Y = Y.long()
    a = time()
    Z = torch.mm(X, Y)
    return time() - a


def mm_float(X, Y):
    X = X.float()
    Y = Y.float()
    a = time()
    Z = torch.mm(X, Y)
    return time() - a


i8, i16, f = 0, 0, 0
for _ in range(n_test):
    X = torch.randint(100, size=(SIZE, SIZE))
    Y = torch.randint(100, size=(SIZE, SIZE))
    i8 += mm_int8(X, Y)
    #i16 += mm_int16(X, Y)
    #f += mm_float(X, Y)
for _ in range(n_test):
    X = torch.randn(SIZE, SIZE)*100
    Y = torch.randn(SIZE, SIZE)*100
    f += mm_float(X, Y)

print("int8")
print(i8 / n_test)

# print("int16")
# print(i16 / n_test)

print("float32")
print(f / n_test)
