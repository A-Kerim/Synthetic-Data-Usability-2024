import torchvision.models as models
# pip install --upgrade torch torchvision (to access ViT)
import warnings
import torch.nn as nn


warnings.filterwarnings("ignore")


def AlexNet(ds_name, num_classes, isFnT):
    print("AlexNet")
    img_size = 320
    if isFnT:
        model = models.alexnet(pretrained=True)
        for param in model.features.parameters():
            param.requires_grad = False
    else:

        model = models.alexnet(pretrained=False) #for metrics comparision

    model.classifier[6] = nn.Linear(4096, num_classes)

    loss_fn = nn.CrossEntropyLoss()
    return model, loss_fn, img_size


def VGG(ds_name, num_classes,isFnT):
    print("VGG")
    img_size = 224

    if isFnT:
        model = models.vgg19_bn(pretrained=True)
        for param in model.features.parameters():
            param.requires_grad = False
    else:
        model = models.vgg19_bn(pretrained=False)#for metrics comparision

    model.classifier[6] = nn.Linear(4096, num_classes)

    loss_fn = nn.CrossEntropyLoss()
    return model, loss_fn, img_size


def REGNet(ds_name, num_classes, isFnT):
    print("REGNet")
    img_size = 320
    if isFnT:
        model = models.regnet_x_800mf(pretrained=True)
        for param in model.parameters():
            param.requires_grad = False
    else:
        model = models.regnet_x_800mf(pretrained=False)#for metrics comparision

    model.fc = nn.Linear(672,num_classes)

    loss_fn = nn.CrossEntropyLoss()
    return model, loss_fn, img_size


def SwinTransformer(ds_name, num_classes, isFnT):
    print("SwinTransformer")
    img_size = 320

    if isFnT:
        model = models.swin_v2_b(pretrained=True)
        for param in model.features.parameters():
            param.requires_grad = False
    else:
        model = models.swin_v2_b(pretrained=False)#for metrics comparision

    model.head = nn.Linear(1024,num_classes)

    loss_fn = nn.CrossEntropyLoss()
    return model, loss_fn, img_size


def EfficientNet_b0(ds_name, num_classes, isFnT):
    print("EfficientNet_b0")

    if isFnT:
        model = models.efficientnet_b0(pretrained=True)
        # Freeze all base layers in the "features" section of the model (the feature extractor) by setting requires_grad=False
        for param in model.features.parameters():
            param.requires_grad = False
    else:
        model = models.efficientnet_b0(pretrained=False)#for metrics comparision

    img_size = 320

    # model.classifier = nn.Sequential(
    #     nn.Dropout(p=0.2, inplace=True),
    #     nn.Linear(in_features=1280, out_features=num_classes, bias=True))

    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)

    loss_fn = nn.CrossEntropyLoss()

    return model, loss_fn, img_size


def ViT(ds_name, num_classes, isFnT):
    print("ViT")

    if isFnT:
        model = models.vit_b_16(pretrained=True)
        # Freeze all base layers in the "features" section of the model (the feature extractor) by setting requires_grad=False
        for param in model.parameters():
            param.requires_grad = False
    else:
        model = models.vit_b_16(pretrained=False)#for metrics comparision

    img_size = 224
    model.heads = nn.Linear(in_features=768, out_features=num_classes, bias=True)


    loss_fn = nn.CrossEntropyLoss()

    return model, loss_fn, img_size

def ViT_scratch(ds_name, num_classes):
    print("ViT_MyApproach (with 4 (Linear and Dropout) layers)")
    model = models.vit_b_16(pretrained=True)
    img_size = 224

    # Freeze all base layers in the "features" section of the model (the feature extractor) by setting requires_grad=False
    for param in model.parameters():
        param.requires_grad = False

    model.heads = nn.Sequential(
        nn.Dropout(p=0.2, inplace=True),
        nn.Linear(in_features=768, out_features=384, bias=True),
        nn.Dropout(p=0.2, inplace=True),
        nn.Linear(in_features=384, out_features=192, bias=True),
        nn.Dropout(p=0.2, inplace=True),
        nn.Linear(in_features=192, out_features=32, bias=True),
        nn.Dropout(p=0.2, inplace=True),
        nn.Linear(in_features=32, out_features=num_classes, bias=True))

    loss_fn = nn.CrossEntropyLoss()

    return model, loss_fn, img_size

def ImagiVQA(ds_name, num_classes):
    print("ImagiVQA")
    model = models.vit_b_16(pretrained=True)
    img_size = 224

    # Freeze all base layers in the "features" section of the model (the feature extractor) by setting requires_grad=False
    for param in model.parameters():
        param.requires_grad = False

    model.heads = nn.Sequential(
        nn.Dropout(p=0.2, inplace=True),
        nn.Linear(in_features=768, out_features=384, bias=True),
        nn.Dropout(p=0.2, inplace=True),
        nn.Linear(in_features=384, out_features=192, bias=True),
        nn.Dropout(p=0.2, inplace=True),
        nn.Linear(in_features=192, out_features=32, bias=True),
        nn.Dropout(p=0.2, inplace=True),
        nn.Linear(in_features=32, out_features=num_classes, bias=True))


    loss_fn = nn.CrossEntropyLoss()

    return model, loss_fn, img_size