from typing import List, Type, Optional
from .rede import RedeBase
import torch.nn as nn
import torchvision.models as models


class RedeFactory:
    @staticmethod
    def createRede(
        sChannels: int,
        sOutput: int,
        sLayers: List[int],
        sKernel: int,
        fnActivation: Type[nn.Module],
        dropout: bool = False,
        base_model: Optional[str] = None,
    ) -> RedeBase:
        rede = RedeBase()

        if base_model:
            if base_model.lower() == "vgg":
                pretrained_model = models.vgg16(pretrained=True)
                for param in pretrained_model.features.parameters():
                    param.requires_grad = False  # Freeze weights
                layers = list(pretrained_model.features.children())
                rede.layers.extend(layers)
                input_features = pretrained_model.classifier[0].in_features
            elif base_model.lower() == "resnet":
                pretrained_model = models.resnet18(pretrained=True)
                for param in pretrained_model.parameters():
                    param.requires_grad = False
                layers = list(pretrained_model.children())[:-1]
                rede.layers.extend(layers)
                rede.layers.append(nn.Flatten())
                input_features = pretrained_model.fc.in_features
            else:
                raise ValueError(f"Unsupported base_model: {base_model}")

            rede.layers.append(nn.Linear(input_features, 120))
            rede.layers.append(fnActivation())
            if dropout:
                rede.layers.append(nn.Dropout(0.25))
            rede.layers.append(nn.Linear(120, 84))
            rede.layers.append(fnActivation())
            rede.layers.append(nn.Linear(84, sOutput))
        else:
            sAntes = sChannels
            sLinearInput = 32
            for sLayer in sLayers:
                rede.layers.append(nn.Conv2d(sAntes, sLayer, sKernel))
                rede.layers.append(fnActivation())
                sLinearInput = (sLinearInput - sKernel) + 1
                if dropout:
                    rede.layers.append(nn.Dropout(0.25))
                sAntes = sLayer

            sLinearInput = sLinearInput**2 * sAntes
            rede.layers.append(nn.Flatten())
            rede.layers.append(nn.Linear(sLinearInput, 120))
            rede.layers.append(fnActivation())
            rede.layers.append(nn.Linear(120, 84))
            rede.layers.append(fnActivation())
            rede.layers.append(nn.Linear(84, sOutput))

        return rede
