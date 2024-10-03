import argparse
import configparser
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
import pandas as pd


class NeuralNetwork(nn.Module):
    def __init__(self, vocab_size: int, num_class: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(vocab_size, 1024),
            nn.GELU(),
            nn.Linear(1024, 2048),
            nn.GELU(),
            nn.Linear(2048, 1024),
            nn.GELU(),
            nn.Linear(1024, 512),
            nn.GELU(),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Linear(256, num_class),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class Data(Dataset):
    def __init__(self, path: str) -> None:
        df = pd.read_csv(path)
        self.labels = torch.tensor(df["doc_categoria"].values, dtype=torch.long)
        self.features = torch.tensor(df.iloc[:, :-1].values, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.features[idx], self.labels[idx]


def train(
    model: nn.Module,
    fnLoss: nn.Module,
    optimizer: Optimizer,
    dataloader: DataLoader,
    testDataloader: DataLoader,
    epochs: int,
    device: torch.device,
) -> None:

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for _, (inputs, targets) in enumerate(dataloader):
            optimizer.zero_grad()
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = fnLoss(outputs, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch: {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

        accuracy = evaluate(model, testDataloader, device)
        print(f"Epoch {epoch+1}/{epochs}, Test Accuracy: {accuracy:.2f}%")


def evaluate(model: nn.Module, dataloader: DataLoader, device) -> float:
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for _, (inputs, targets) in enumerate(dataloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
    accuracy = 100 * correct / total
    return accuracy


def main() -> None:
    parser = argparse.ArgumentParser(description="Treinamento MLP")
    parser.add_argument(
        "--config",
        type=str,
        default="configurations.ini",
        help="PATH para configurações de treino",
    )

    args = parser.parse_args()
    config = configparser.ConfigParser()
    config.read(args.config)

    batch_size = config.getint("TRAINING", "batch_size")
    epochs = config.getint("TRAINING", "epochs")
    learning_rate = config.getfloat("TRAINING", "learning_rate")
    use_cuda = config.getboolean("TRAINING", "cuda")

    cuda = use_cuda and torch.cuda.is_available()

    if cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    train_dataset = Data("data/noticias_final_tfidf_df_300.csv")
    # gerar dados treino
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    model = NeuralNetwork(300, 2).to(device)

    loss = nn.CrossEntropyLoss()

    print(f"Learning Rate: {learning_rate}")

    print(model)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train(model, loss, optimizer, train_dataloader, test_dataloader, epochs, device)

    torch.save(model.state_dict(), "model.pth")


if __name__ == "__main__":
    main()
