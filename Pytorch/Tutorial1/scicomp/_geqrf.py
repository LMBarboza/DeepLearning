import torch
import numpy as np
import time

N: int = 10000


if torch.cuda.is_available():
    DEVICE = torch.device("cuda")


def main() -> None:
    A: np.ndarray = np.random.randn(N, N)
    start: float = time.time()
    np.linalg.qr(A)
    end: float = time.time()
    print(f"Numpy: {end - start:.4f} s")

    A_torch: torch.Tensor = torch.randn(N, N, device=DEVICE)
    torch.cuda.synchronize()
    start: float = time.time()
    torch.linalg.qr(A_torch)
    torch.cuda.synchronize()
    end: float = time.time()
    print(f"Pytorch: {end - start:.4f} s")


if __name__ == "__main__":
    main()
