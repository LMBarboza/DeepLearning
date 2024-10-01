import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.optim.optimizer import Optimizer


def gen_data(n: int = 100, noise: float = 0.2) -> tuple:
    x = np.linspace(0, 10, n)
    y = 2.5 * x + 1.0 + noise * np.random.randn(n)
    return x, y


# def gen_data(n: int = 100, features: int = 3, noise: float = 0.2) -> tuple:
#    X = np.random.randn(n, features)
#    w = np.array([1.5, -2.0, 3.0])
#    y = X @ w + 1.0 + noise * np.random.randn(n)
#    return X, y


class LinReg(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, x) -> torch.Tensor:
        return self.linear(x)


def train_model(
    model: nn.Module,
    fnLoss: nn.Module,
    optimizer: Optimizer,
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    epochs: int = 500,
) -> None:
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = fnLoss(outputs, y_train)
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 100 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")


def plot(x, y, y_pred):
    plt.scatter(x, y, label="Groung Truth")
    plt.plot(x, y_pred, label="Model Pred", color="red")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Regressão Linear Simples")
    plt.legend()
    plt.show()


def main():
    print("Treinando Regressão Simples...")
    x, y = gen_data(n=100, noise=0.5)

    x_tensor = torch.tensor(x, dtype=torch.float32).view(-1, 1)
    y_tensor = torch.tensor(y, dtype=torch.float32).view(-1, 1)

    model = LinReg()

    print(model)

    loss = nn.MSELoss()

    optimizer = optim.SGD(model.parameters(), lr=0.01)
    train_model(model, loss, optimizer, x_tensor, y_tensor, epochs=500)

    with torch.no_grad():
        y_pred = model(x_tensor).numpy()

    # plot(x_tensor.numpy(), y_tensor.numpy(), y_pred)


if __name__ == "__main__":
    main()
