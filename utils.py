from math import ceil, floor, log
import torch


def i_poly(q, S, a, b, c):
    q_b = floor(b/S)
    S_out = a*S*S
    q_c = floor(c/S_out)
    q_l = q.long()
    q_out = (q_l + q_b)*(q_l+q_b)+q_c
    return q_out, S_out


def i_exp(q, S):
    a, b, c = 0.3585, 1.353, 0.344
    q_ln = floor(log(2)/S)
    z = -torch.div(q.long(), q_ln, rounding_mode="floor")
    q_p = q+z*q_ln
    q_o, S_o = i_poly(q_p, S, a, b, c)
    q_o >>= z
    return q_o, S_o


def i_sqrt(n, n_bits):
    x = 2 << ceil(n_bits // 2)
    for i in range(10):
        x = (x + torch.div(n, x, rounding_mode="floor"))
        x >>= 1
    return x


if __name__ == "__main__":
    x = torch.rand(5)*4-2
    print(x)
    scale = 4 / 256
    q = (x / scale).char()
    print(q)
    print(i_sqrt(q.abs(), 8))
