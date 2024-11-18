import argparse
import json
import configparser
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from modules.rede_factory import RedeFactory
from modules.trainer import Trainer
from modules.treino_strategy import STDStrategy


def main() -> None:
    parser = argparse.ArgumentParser(description="Transfer Learning")
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
    learning_rate = config.getfloat("TRAINING", "learning_rate")
    epochs = config.getint("TRAINING", "epochs")
    use_cuda = config.getboolean("TRAINING", "cuda")
    data = config.get("TRAINING", "dataset")
    base_model = config.get("TRAINING", "base")

    cuda = use_cuda and torch.cuda.is_available()

    if cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    match data:
        case "wow":

            transform = transforms.Compose(
                [
                    transforms.Resize((256, 384)),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
                    ),
                ]
            )
            dataset = datasets.ImageFolder(
                root="./data/Wonders of World/Wonders of World/", transform=transform
            )

            train_size = int(0.8 * len(dataset))
            test_size = len(dataset) - train_size
            train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

            train_dataloader = DataLoader(
                train_dataset, batch_size=batch_size, shuffle=True
            )
            test_dataloader = DataLoader(
                test_dataset, batch_size=batch_size, shuffle=False
            )

        case "xray":
            transform = transforms.Compose(
                [
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ]
            )
            train_dataset = datasets.ImageFolder(
                root="./data/chest_xray/train/", transform=transform
            )
            test_dataset = datasets.ImageFolder(
                root="./data/chest_xray/test/", transform=transform
            )

            train_dataloader = DataLoader(
                train_dataset, batch_size=batch_size, shuffle=True
            )
            test_dataloader = DataLoader(
                test_dataset, batch_size=batch_size, shuffle=False
            )

    sChannels = 3
    sKernel = 5
    sOutput = 10

    accuracy_list = []
    hiddenLayers = [32, 64, 128]

    model = RedeFactory.createRede(
        sChannels, sOutput, hiddenLayers, sKernel, fnActivation=nn.ReLU
    ).to(device)

    print(model)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    loss = nn.CrossEntropyLoss()

    strategy = STDStrategy(loss, optimizer)

    trainer = Trainer(model, strategy)
    accuracy = trainer.train(train_dataloader, test_dataloader, epochs, device)
    accuracy_list.append(accuracy)
    torch.save(model.state_dict(), "models/model.pth")

    with open("results/layers_accuracy.json", "w") as f:
        json.dump(accuracy_list, f)


if __name__ == "__main__":
    main()
