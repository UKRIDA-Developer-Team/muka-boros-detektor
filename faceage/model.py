import torch
from torch import nn
from torchvision import models, transforms

MEAN = [0.485,0.456,0.406]
STD = [0.229,0.224,0.225]
PREPROCESS = {'resize':256, 'crop':224, 'mean':MEAN, 'std':STD}


def make_transform(training=False, augmentation=True):
    operations = [transforms.Resize(256), transforms.CenterCrop(224)]
    if training and augmentation:
        operations += [transforms.RandomHorizontalFlip(),
                       transforms.ColorJitter(brightness=0.1,contrast=0.1)]
    return transforms.Compose(operations + [transforms.ToTensor(),transforms.Normalize(MEAN,STD)])


def create_model(weights='IMAGENET1K_V1', train_median=0.0):
    if weights not in {'IMAGENET1K_V1','none'}: raise ValueError('Bobot pralatih tidak dikenal.')
    source = models.MobileNet_V3_Small_Weights.IMAGENET1K_V1 if weights!='none' else None
    model = models.mobilenet_v3_small(weights=source)
    previous = model.classifier[-1]
    model.classifier[-1] = nn.Linear(previous.in_features,1)
    nn.init.constant_(model.classifier[-1].bias,float(train_median))
    return model


def configure_training(model, method, unfreeze_blocks=3):
    if method not in {'b1','m1','b1_control'}: raise ValueError('Metode jaringan tidak dikenal.')
    if not isinstance(unfreeze_blocks,int) or not 1<=unfreeze_blocks<=len(model.features):
        raise ValueError('Jumlah modul fine tuning harus berada dalam rentang backbone.')
    for parameter in model.parameters(): parameter.requires_grad = False
    for parameter in model.classifier.parameters(): parameter.requires_grad = True
    if method=='m1':
        for layer in model.features[-unfreeze_blocks:]:
            for parameter in layer.parameters(): parameter.requires_grad = True


def set_training_mode(model, method, unfreeze_blocks=3):
    model.train()
    model.features.eval()
    if method=='m1':
        for layer in model.features[-unfreeze_blocks:]: layer.train()


def resolve_device(request='auto'):
    if request not in {'auto','cpu','cuda'}: raise ValueError('Device harus auto, cpu, atau cuda.')
    if request=='cuda' and not torch.cuda.is_available(): raise ValueError('GPU CUDA tidak tersedia.')
    return torch.device('cuda' if request=='cuda' or (request=='auto' and torch.cuda.is_available()) else 'cpu')
