import torch

def com_mult(a, b):
    r1, i1 = a.real, a.imag
    r2, i2 = b.real, b.imag
    real = r1 * r2 - i1 * i2
    imag = r1 * i2 + i1 * r2
    return torch.complex(real, imag)

def conj(a):
    a.imag = -a.imag
    return a

def ccorr(a, b):
    return torch.fft.irfft(com_mult(conj(torch.fft.rfft(a)), torch.fft.rfft(b)), a.shape[-1])