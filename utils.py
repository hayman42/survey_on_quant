from math import ceil, log, floor
import torch
import torch.nn.functional as F


def i_poly(q, S, a, b, c):
    q_b = floor(b/S)
    S_out = a*S*S
    q_c = floor(c/S_out)
    q_out = (q + q_b)*(q+q_b)+q_c
    return q_out, S_out


def i_exp(q, S):
    a, b, c = 0.3585, 1.353, 0.344
    q_ln = floor(log(2)/S)
    z = -(q / q_ln).floor()
    q_p = q+z*q_ln
    q_o, S_o = i_poly(q_p, S, a, b, c)
    q_o >>= z
    return q_o, S_o


def i_sqrt(n, n_bits):
    x = 2 << ceil(n_bits // 2)
    for i in range(10):
        x = (x + (n / x)/floor())
        x >>= 1
    return x


def i_softmax(q, S):
    q_exp, S_exp = i_exp(q, S)
    q_sum = q_exp.sum(dim=1)
    return q_exp/q_sum


if __name__ == "__main__":
    x = torch.rand(5, 5)*4-2
    scale = 4 / 256
    q = (x / scale).floor()
    print(F.softmax(x, dim=1))
    print(i_softmax(q, scale))
