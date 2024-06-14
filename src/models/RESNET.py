import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet18_Weights


def setup_resnet18(device):
    # Load the pretrained ResNet18 model
    model = models.resnet18(weights=ResNet18_Weights.DEFAULT)

    # Move the model to the appropriate device (GPU if available, otherwise CPU)
    if torch.cuda.is_available():
        model.cuda()
    else:
        model.to(device)

    # Get the number of features in the last fully connected layer
    num_ftrs = model.fc.in_features

    # Modify the fully connected layer to output a single value (for occlusion score)
    model.fc = nn.Linear(num_ftrs, 1).to(device)

    return model
