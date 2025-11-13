# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# from torchvision_wj.models.segwithbox.boosting_ensemble import (
#     BoostingDeeplabUnetResNet,
#     BoostingResNetUnetDeeplab,
#     BoostingUnetDeeplabResNet,
# )

# # try:
# from .enet_23 import ENet as UNet
# from .residualunet import ResidualUNet as ResNetSegmentation
# from .DeepLabv3 import DeepLabV3

# # except ImportError:
# # from .simple_models import (
# #     SimpleUNet as UNet,
# #     SimpleDeepLabV3 as DeepLabV3,
# #     SimpleResNet as ResNetSegmentation,
# # )

# # from .simple_models import SimpleResNet as ResNetSegmentation
# # from .simple_models import SimpleDeepLabV3 as DeepLabV3


# def create_boosting_ensemble_from_scratch(
#     in_dim: int,
#     out_dim: int,
#     softmax: bool,
#     configuration: int = 1,
#     model_type: str = "standard",
# ):
#     """
#     Factory function that creates complete boosting ensembles from scratch

#     This function handles the internal creation of all three models (UNet, ResNet, DeepLabV3)
#     and arranges them in the specified boosting configuration.

#     Args:
#         in_dim: Number of input channels (1 for grayscale, 3 for RGB)
#         out_dim: Number of output classes/channels
#         softmax: Whether to use softmax activation (True for multi-class, False for binary)
#         configuration: Which boosting configuration to use:
#             1: UNet -> DeepLabV3 -> ResNet
#             2: ResNet -> UNet -> DeepLabV3
#             3: DeepLabV3 -> UNet -> ResNet
#         model_type: Type of models to create ("standard", "lightweight", "heavy")

#     Returns:
#         Initialized boosting model instance

#     Raises:
#         ValueError: If invalid configuration or model_type is provided
#     """

#     # Create individual models based on model_type
#     if model_type == "lightweight":
#         # Smaller models for faster training/inference
#         unet_channels = 16
#         resnet_backbone = "resnet18"
#         deeplab_channels = 32
#     elif model_type == "heavy":
#         # Larger models for better performance
#         unet_channels = 64
#         resnet_backbone = "resnet50"
#         deeplab_channels = 128
#     else:  # standard
#         # Balanced models
#         unet_channels = 32
#         resnet_backbone = "resnet34"
#         deeplab_channels = 64

#     print(f"Creating {model_type} models with configuration {configuration}")

#     # Create all three models
#     unet_model = UNet(
#         in_dim=in_dim,
#         out_dim=out_dim,
#         softmax=softmax,
#         # channels_in=unet_channels
#     )
#     resnet_model = ResNetSegmentation(
#         in_dim=in_dim,
#         out_dim=out_dim,
#         softmax=softmax,
#         backbone=resnet_backbone,
#         # channels_in=32,
#     )
#     deeplab_model = DeepLabV3(
#         in_dim=in_dim,
#         out_dim=out_dim,
#         softmax=softmax,
#         # channels_in=deeplab_channels
#     )

#     # Create the appropriate boosting configuration
#     if configuration == 1:
#         return BoostingUnetDeeplabResNet(
#             unet_model, deeplab_model, resnet_model, in_dim, out_dim, softmax
#         )
#     elif configuration == 2:
#         return BoostingResNetUnetDeeplab(
#             resnet_model, unet_model, deeplab_model, in_dim, out_dim, softmax
#         )
#     elif configuration == 3:
#         return BoostingDeeplabUnetResNet(
#             deeplab_model, unet_model, resnet_model, in_dim, out_dim, softmax
#         )
#     else:
#         raise ValueError(f"Invalid configuration: {configuration}. Use 1, 2, or 3.")


# def create_boosting_with_pretrained(
#     in_dim: int,
#     out_dim: int,
#     softmax: bool,
#     configuration: int = 1,
#     pretrained: bool = True,
#     freeze_backbone: bool = False,
# ):
#     """
#     Factory function that creates boosting ensembles with optional pretrained weights

#     Args:
#         in_dim: Number of input channels
#         out_dim: Number of output classes/channels
#         softmax: Whether to use softmax activation
#         configuration: Boosting configuration (1, 2, or 3)
#         pretrained: Whether to use pretrained weights for ResNet/DeepLabV3
#         freeze_backbone: Whether to freeze backbone weights during training

#     Returns:
#         Initialized boosting model with pretrained weights if requested
#     """

#     # Create models with pretrained options
#     unet_model = UNet(in_dim=in_dim, out_dim=out_dim, softmax=softmax)

#     resnet_model = ResNetSegmentation(
#         in_dim=in_dim,
#         out_dim=out_dim,
#         softmax=softmax,
#         pretrained=pretrained,
#         freeze_backbone=freeze_backbone,
#     )

#     deeplab_model = DeepLabV3(
#         in_dim=in_dim,
#         out_dim=out_dim,
#         softmax=softmax,
#         pretrained=pretrained,
#         freeze_backbone=freeze_backbone,
#     )

#     # Create boosting configuration
#     if configuration == 1:
#         return BoostingUnetDeeplabResNet(
#             unet_model, deeplab_model, resnet_model, in_dim, out_dim, softmax
#         )
#     elif configuration == 2:
#         return BoostingResNetUnetDeeplab(
#             resnet_model, unet_model, deeplab_model, in_dim, out_dim, softmax
#         )
#     elif configuration == 3:
#         return BoostingDeeplabUnetResNet(
#             deeplab_model, unet_model, resnet_model, in_dim, out_dim, softmax
#         )
#     else:
#         raise ValueError(f"Invalid configuration: {configuration}")


# def create_custom_boosting(
#     in_dim: int, out_dim: int, softmax: bool, configuration: int = 1, **model_kwargs
# ):
#     """
#     Factory function with customizable model parameters

#     Args:
#         in_dim: Number of input channels
#         out_dim: Number of output classes/channels
#         softmax: Whether to use softmax activation
#         configuration: Boosting configuration (1, 2, or 3)
#         **model_kwargs: Custom parameters for each model type:
#             - unet_channels: Base channels for UNet
#             - resnet_backbone: Backbone for ResNet ('resnet18', 'resnet34', 'resnet50')
#             - deeplab_channels: Output channels for DeepLabV3
#             - pretrained: Whether to use pretrained weights
#             - dropout_rate: Dropout rate for all models

#     Returns:
#         Customized boosting model instance
#     """

#     # Extract custom parameters with defaults
#     unet_channels = model_kwargs.get("unet_channels", 32)
#     resnet_backbone = model_kwargs.get("resnet_backbone", "resnet34")
#     deeplab_channels = model_kwargs.get("deeplab_channels", 64)
#     pretrained = model_kwargs.get("pretrained", True)
#     dropout_rate = model_kwargs.get("dropout_rate", 0.1)

#     print(f"Creating custom boosting ensemble:")
#     print(f"  - UNet channels: {unet_channels}")
#     print(f"  - ResNet backbone: {resnet_backbone}")
#     print(f"  - DeepLabV3 channels: {deeplab_channels}")
#     print(f"  - Pretrained: {pretrained}")
#     print(f"  - Configuration: {configuration}")

#     # Create customized models
#     unet_model = UNet(
#         in_dim=in_dim,
#         out_dim=out_dim,
#         softmax=softmax,
#         channels_in=unet_channels,
#         dropout_rate=dropout_rate,
#     )

#     resnet_model = ResNetSegmentation(
#         in_dim=in_dim,
#         out_dim=out_dim,
#         softmax=softmax,
#         backbone=resnet_backbone,
#         pretrained=pretrained,
#     )

#     deeplab_model = DeepLabV3(
#         in_dim=in_dim,
#         out_dim=out_dim,
#         softmax=softmax,
#         channels_in=deeplab_channels,
#         pretrained=pretrained,
#     )

#     # Create boosting configuration
#     if configuration == 1:
#         return BoostingUnetDeeplabResNet(
#             unet_model, deeplab_model, resnet_model, in_dim, out_dim, softmax
#         )
#     elif configuration == 2:
#         return BoostingResNetUnetDeeplab(
#             resnet_model, unet_model, deeplab_model, in_dim, out_dim, softmax
#         )
#     elif configuration == 3:
#         return BoostingDeeplabUnetResNet(
#             deeplab_model, unet_model, resnet_model, in_dim, out_dim, softmax
#         )
#     else:
#         raise ValueError(f"Invalid configuration: {configuration}")


# # Usage examples
# if __name__ == "__main__":
#     print("=== Boosting Ensemble Factory Demo ===")

#     # Example 1: Standard configuration
#     print("\n1. Standard UNet->DeepLabV3->ResNet:")
#     model1 = create_boosting_ensemble_from_scratch(
#         in_dim=1, out_dim=2, softmax=True, configuration=1, model_type="standard"
#     )

#     # Example 2: Lightweight with pretrained weights
#     print("\n2. Lightweight ResNet->UNet->DeepLabV3:")
#     model2 = create_boosting_with_pretrained(
#         in_dim=3,
#         out_dim=5,
#         softmax=True,
#         configuration=2,
#         pretrained=True,
#         freeze_backbone=True,
#     )

#     # Example 3: Custom configuration
#     print("\n3. Custom DeepLabV3->UNet->ResNet:")
#     model3 = create_custom_boosting(
#         in_dim=1,
#         out_dim=2,
#         softmax=True,
#         configuration=3,
#         unet_channels=64,
#         resnet_backbone="resnet50",
#         deeplab_channels=128,
#         pretrained=True,
#         dropout_rate=0.2,
#     )

#     # Test the models
#     test_input = torch.randn(2, 1, 256, 256)

#     print(f"\nTesting model 1...")
#     output1 = model1(test_input)
#     print(f"Output shape: {output1[0].shape}")

#     print(
#         f"\nTotal parameters model 1: {sum(p.numel() for p in model1.parameters()):,}"
#     )
