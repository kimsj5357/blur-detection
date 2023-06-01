import torch
from torch import nn

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(3*512*512, 64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.1),
            nn.Linear(64, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.1),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.model(input.view(1, -1))

    def loss(self, img_D, label):
        adversarial_loss = nn.BCELoss()
        loss = adversarial_loss(img_D, label)
        return loss

    def get_optimizer_params(self, lr, weight_decay):
        params = []
        for key, value in dict(self.named_parameters()).items():
            if value.requires_grad:
                params += [{'params': [value], 'lr': lr, 'weight_decay': weight_decay}]
        return params