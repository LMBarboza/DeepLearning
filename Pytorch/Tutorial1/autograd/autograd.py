import torch
from typing import Callable


def f(x: torch.Tensor) -> torch.Tensor:
    return x**3


def g(x: torch.Tensor) -> torch.Tensor:
    return x**2 + 3 * x + 1


def derivative(f: Callable[[float], float], x: float, h: float = 1e-5) -> float:
    return (f(x + h) - f(x - h)) / (2 * h)


def main() -> None:
    print("Exemplo 1: f(x) = x^3")

    x = torch.tensor(2.0, requires_grad=True)

    y = f(x)

    print(x)
    print(y)
    print(f"y: {y.item()}")

    y.backward()

    print(f"Grad: {x.grad.item()}")

    print("Exemplo 2: g(x) = x^2 + 3x + 1")

    x = torch.tensor(2.0, requires_grad=True)

    y = g(x)
    print(f"y: {y.item()}")

    y.backward()
    print(f"Grad: {x.grad.item()}")

    print("Usando grad() no exemplo 1")

    x = torch.tensor(2.0, requires_grad=True)

    y = f(x)

    grad_x = torch.autograd.grad(outputs=y, inputs=x)[0]
    print(f"Grad: {grad_x.item()}")

    print("Exemplo 4: Diferenças Finitas")

    def h(x: float) -> float:
        return x**3

    x_val = 2.0
    numerical_grad = derivative(h, x_val)
    print(f"dy/dx : {numerical_grad}")


if __name__ == "__main__":
    main()
