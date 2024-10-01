import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.optim.optimizer import Optimizer


def gen_data(n: int = 100, features: int = 3, noise: float = 0.2) -> tuple:
    X = np.random.randn(n, features)
    w = np.array([1.5, -2.0, 3.0])
    y = (X @ w + 1.0 + noise * np.random.randn(n) > 0).astype(int)
    return X, y


class LogReg(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(3, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x) -> torch.Tensor:
        x = self.linear(x)
        x = self.sigmoid(x)
        return x


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


def main():
    print("Treinando Regressão Simples...")
    x, y = gen_data()

    x_tensor = torch.tensor(x, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32).view(-1, 1)

    model = LogReg()

    print(model)

    loss = nn.BCELoss()

    optimizer = optim.SGD(model.parameters(), lr=0.01)
    train_model(model, loss, optimizer, x_tensor, y_tensor, epochs=500)

    with torch.no_grad():
        y_pred = model(x_tensor).numpy()

    # plot(x_tensor.numpy(), y_tensor.numpy(), y_pred)


if __name__ == "__main__":
    main()
