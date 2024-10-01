import torch
import torch.nn as nn
from torch.nn.modules import ReLU
import torch.optim as optim
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# usar exemplo sem aceleração por GPU
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")


class Autoencoder(nn.Module):
    def __init__(self, latent_dim: int = 10) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(28 * 28, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, latent_dim),
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 28 * 28),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_encoded = self.encoder(x)
        x_decoded = self.decoder(x_encoded)
        return x_decoded


def train(
    model: nn.Module,
    fnLoss: nn.Module,
    optimizer: Optimizer,
    dataloader: DataLoader,
    testDataloader: DataLoader,
    epochs: int,
) -> None:

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for _, (inputs, targets) in enumerate(dataloader):
            optimizer.zero_grad()
            inputs = inputs.view(-1, 28 * 28)
            outputs = model(inputs)
            loss = fnLoss(outputs, inputs)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch: {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")


def main() -> None:
    batch_size = 64
    epochs = 10
    learning_rate = 0.01

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )

    train_dataset = datasets.MNIST(
        root="./data", train=True, download=True, transform=transform
    )

    test_dataset = datasets.MNIST(
        root="./data", train=False, download=True, transform=transform
    )

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    model = Autoencoder()

    loss = nn.MSELoss()

    print(f"Learning Rate: {learning_rate}")

    print(model)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train(model, loss, optimizer, train_dataloader, test_dataloader, epochs)

    # salvar parametros do modelo
    torch.save(model.state_dict(), "model.pth")


if __name__ == "__main__":
    main()
