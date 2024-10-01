import torch
import numpy as np
import time

N: int = 1000


if torch.cuda.is_available():
    DEVICE = torch.device("cuda")


def _gemm(A: list[list[float]], B: list[list[float]], C: list[list[float]]) -> None:
    """Multiplicação Matricial Usando Tipos Básicos do Python e laços em Python"""
    for i in range(len(A)):
        for j in range(len(B[0])):
            for k in range(len(B)):
                C[i][j] += A[i][k] * B[k][j]


def main() -> None:
    A: np.ndarray = np.random.randn(N, N)
    B: np.ndarray = np.random.randn(N, N)
    C: np.ndarray = np.zeros((N, N))

    start: float = time.time()
    _gemm(A.tolist(), B.tolist(), C.tolist())
    end: float = time.time()
    print(f"Python: {end - start:.4f} s")

    start: float = time.time()
    res: np.ndarray = np.matmul(A, B)
    end: float = time.time()
    print(f"Numpy: {end - start:.4f} s")

    A_torch: torch.tensor = torch.randn(N, N, device=DEVICE)
    B_torch: torch.tensor = torch.randn(N, N, device=DEVICE)
    torch.cuda.synchronize()
    start: float = time.time()
    C_torch: torch.tensor = torch.matmul(A_torch, B_torch)
    torch.cuda.synchronize()
    end: float = time.time()
    print(f"Pytorch: {end - start:.4f} s")


if __name__ == "__main__":
    main()
