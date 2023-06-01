from torch import nn
from torchvision.models import vgg16

from config import opt

def vgg_16(device):
    model = vgg16(pretrained=True).to(device)

    features = list(model.features)[:30]
    for layer in features[:10]:
        for p in layer.parameters():
            p.requires_grad = False
    extractor = nn.Sequential(*features)

    classifier = list(model.classifier)
    del classifier[6]
    if not opt.use_drop:
        del classifier[5]
        del classifier[2]
    classifier = nn.Sequential(*classifier)

    return extractor, classifier
