from torchvision_wj.models.segwithbox.SequentialBoostingEnsemble import (
    SequentialBoostingEnsemble,
)
from torchvision_wj.models.segwithbox.boosting import create_boosted_model
from torchvision_wj.models.segwithbox.boosting_ensemble import (
    create_boosting_ensemble_fixed,
)
from torchvision_wj.models.segwithbox.boosting_ensemble2 import (
    create_boosting_ensemble_complete,
)
from torchvision_wj.models.segwithbox.boosting_v3 import create_sequential_boosting_v3

# from torchvision_wj.models.segwithbox.boosting_v4 import (
# )
from torchvision_wj.models.segwithbox.residualunet import ResidualUNet
from torchvision_wj.models.segwithbox.enet import ENet
from torchvision_wj.models.segwithbox.enet_23 import ENet as ENet32
from torchvision_wj.models.segwithbox.simple_models import (
    SimpleDeepLabV3,
    SimpleResNet,
    SimpleUNet,
)
from .DeepLabv3 import DeepLabV3

__all__ = [
    "unet_residual",
    "enet",
    "deep_labv3",
    "sequential_boosting_ensembl",
    "boosting",
    "enet32",
    "sequential_boosting_v3",
    "progressive_boosting_v3",
    "adaptive_boosting_v3",
    "simple_deep_labv3",
    "simple_res_net",
    "simple_unet",
    "sequential_boosting_1",
    "sequential_boosting_2",
    "sequential_boosting_3",
    "sequential_boosting_4",
    "sequential_boosting_5",
    "sequential_boosting_6",
]


def unet_residual(input_dim, num_classes, softmax, channels_in=32):
    model = ResidualUNet(input_dim, num_classes, softmax, channels_in)
    return model


def enet(input_dim, num_classes, softmax, channels_in=16):
    model = ENet(input_dim, num_classes, softmax, channels_in)
    return model


def enet32(input_dim, num_classes, softmax, channels_in=16):
    model = ENet32(input_dim, num_classes, softmax, channels_in)
    return model


def deep_labv3(input_dim, num_classes, softmax, channels_in=32):
    model = DeepLabV3(input_dim, num_classes, softmax, channels_in)
    return model


def simple_deep_labv3(input_dim, num_classes, softmax, channels_in=32):
    model = SimpleDeepLabV3(input_dim, num_classes, softmax, channels_in)
    return model


def simple_res_net(input_dim, num_classes, softmax, channels_in=32):
    model = SimpleResNet(input_dim, num_classes, softmax, channels_in)
    return model


def simple_unet(input_dim, num_classes, softmax, channels_in=32):
    model = SimpleUNet(input_dim, num_classes, softmax, channels_in)
    return model


def boosting(input_dim, num_classes, softmax, channels_in=32):
    model = create_boosted_model(input_dim, num_classes, softmax, channels_in)
    return model


def sequential_boosting_v3(input_dim, num_classes, softmax, channels_in=32):
    model = create_sequential_boosting_v3(
        input_dim, num_classes, softmax, channels_in, "sequential"
    )
    return model


def progressive_boosting_v3(input_dim, num_classes, softmax, channels_in=32):
    model = create_sequential_boosting_v3(
        input_dim, num_classes, softmax, channels_in, "progressive"
    )
    return model


def adaptive_boosting_v3(input_dim, num_classes, softmax, channels_in=32):
    model = create_sequential_boosting_v3(
        input_dim, num_classes, softmax, channels_in, "adaptive"
    )
    return model


def __create_models(input_dim, num_classes, softmax, channels_in=32, simple=True):

    if simple:
        unet = simple_unet(input_dim, num_classes, softmax, channels_in=16)
        res_unet = simple_res_net(input_dim, num_classes, softmax, channels_in)
        deeblap = simple_deep_labv3(input_dim, num_classes, softmax, channels_in)

    else:
        unet = ENet32(input_dim, num_classes, softmax, channels_in=16)
        res_unet = ResidualUNet(input_dim, num_classes, softmax, channels_in)
        deeblap = DeepLabV3(input_dim, num_classes, softmax, channels_in)

    return unet, res_unet, deeblap


# TODO first senirio
def sequential_boosting_1(input_dim, num_classes, softmax, channels_in=32, simple=True):
    """
    Unet --->Deeplapv3 ----> ResNet

    Refrence :->

    1. Boosting Unet          --->Deeplapv3.       ----> ResNet
    2. Boosting ResNet.       ---> Unet.           ----> Deeplabv3
    3. Boosting Deeplabv3.    --->Unet.            ---->ResNet

    """
    unet, deeblap, res_unet = __create_models(
        input_dim, num_classes, softmax, channels_in, simple
    )

    return create_boosting_ensemble_fixed(
        model1=unet,
        model2=deeblap,
        model3=res_unet,
        configuration=1,
        model1_type="unet",
        in_dim=input_dim,
        out_dim=num_classes,
        softmax=softmax,
    )


# TODO second senario
def sequential_boosting_2(input_dim, num_classes, softmax, channels_in=32, simple=True):
    """
    ResNet. ---> Unet. ----> Deeplabv3

    Refrence :->

    1. Boosting Unet          --->Deeplapv3.       ----> ResNet
    2. Boosting ResNet.       ---> Unet.           ----> Deeplabv3
    3. Boosting Deeplabv3.    --->Unet.            ---->ResNet

    """
    unet, deeblap, res_unet = __create_models(
        input_dim, num_classes, softmax, channels_in, simple
    )

    return create_boosting_ensemble_fixed(
        model1=res_unet,
        model2=unet,
        model3=deeblap,
        configuration=2,
        model1_type="resnet",
        in_dim=input_dim,
        out_dim=num_classes,
        softmax=softmax,
    )


# TODO third senario
def sequential_boosting_3(input_dim, num_classes, softmax, channels_in=32, simple=True):
    """
    Deeplabv3. ---> Unet. ----> ResNet

    Refrence :->

    1. Boosting Unet          --->Deeplapv3.       ----> ResNet
    2. Boosting ResNet.       ---> Unet.           ----> Deeplabv3
    3. Boosting Deeplabv3.    --->Unet.            ---->ResNet

    """
    unet, deeblap, res_unet = __create_models(
        input_dim, num_classes, softmax, channels_in, simple
    )

    return create_boosting_ensemble_fixed(
        model1=deeblap,
        model2=unet,
        model3=res_unet,
        configuration=3,
        model1_type="deeplab",
        in_dim=input_dim,
        out_dim=num_classes,
        softmax=softmax,
    )


# TODO fourth senario
def sequential_boosting_4(input_dim, num_classes, softmax, channels_in=32, simple=True):
    """
    4. UNet →        ResNet →    DeepLabV3

    Refrence :->

    1. Boosting Unet          --->Deeplapv3.       ----> ResNet
    2. Boosting ResNet.       ---> Unet.           ----> Deeplabv3
    3. Boosting Deeplabv3.    --->Unet.            ---->ResNet

    configurations = [
    4.   (2, "unet",     "UNet →        ResNet →    DeepLabV3"),
    5.   (3, "resnet",   "ResNet →      UNet →      DeepLabV3"),
    6.   (6, "deeplab",  "DeepLabV3 →   ResNet →    UNet"),
    ]

    """
    unet, deeblap, res_unet = __create_models(
        input_dim, num_classes, softmax, channels_in, simple
    )

    return create_boosting_ensemble_complete(
        model1=deeblap,
        model2=unet,
        model3=res_unet,
        configuration=2,
        model1_type="unet",
        in_dim=input_dim,
        out_dim=num_classes,
        softmax=softmax,
    )


# TODO fifth senario
def sequential_boosting_5(input_dim, num_classes, softmax, channels_in=32, simple=True):
    """
    5. ResNet →      UNet →      DeepLabV3

    Refrence :->

    1. Boosting Unet          --->Deeplapv3.       ----> ResNet
    2. Boosting ResNet.       ---> Unet.           ----> Deeplabv3
    3. Boosting Deeplabv3.    --->Unet.            ---->ResNet

    configurations = [
    4.   (2, "unet",     "UNet →        ResNet →    DeepLabV3"),
    5.   (3, "resnet",   "ResNet →      UNet →      DeepLabV3"),
    6.   (6, "deeplab",  "DeepLabV3 →   ResNet →    UNet"),
    ]

    """
    unet, deeblap, res_unet = __create_models(
        input_dim, num_classes, softmax, channels_in, simple
    )

    return create_boosting_ensemble_complete(
        model1=deeblap,
        model2=unet,
        model3=res_unet,
        configuration=3,
        model1_type="resnet",
        in_dim=input_dim,
        out_dim=num_classes,
        softmax=softmax,
    )


# TODO sixth senario
def sequential_boosting_6(input_dim, num_classes, softmax, channels_in=32, simple=True):
    """
    6. Deeplab   ->  ResNet   ->      Unet

    Refrence :->

    1. Boosting Unet          --->Deeplapv3.       ----> ResNet
    2. Boosting ResNet.       ---> Unet.           ----> Deeplabv3
    3. Boosting Deeplabv3.    --->Unet.            ---->ResNet

    configurations = [
    4.   (2, "unet",     "UNet →        ResNet →    DeepLabV3"),
    5.   (3, "resnet",   "ResNet →      UNet →      DeepLabV3"),
    6.   (6, "deeplab",  "DeepLabV3 →   ResNet →    UNet"),
    ]

    """
    unet, deeblap, res_unet = __create_models(
        input_dim, num_classes, softmax, channels_in, simple
    )

    return create_boosting_ensemble_complete(
        model1=deeblap,
        model2=unet,
        model3=res_unet,
        configuration=6,
        model1_type="deeplab",
        in_dim=input_dim,
        out_dim=num_classes,
        softmax=softmax,
    )


def sequential_boosting_ensembl(input_dim, num_classes, softmax, channels_in=32):
    ResidualUNetmodel = ResidualUNet(input_dim, num_classes, softmax, channels_in)
    UNetmodel = ENet(input_dim, num_classes, softmax, channels_in)
    DeepLabV3model = DeepLabV3(input_dim, num_classes, softmax, channels_in)
    model = SequentialBoostingEnsemble(ResidualUNetmodel, UNetmodel, DeepLabV3model)
    return model
