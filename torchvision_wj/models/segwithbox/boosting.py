import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from torchvision_wj.models.segwithbox.DeepLabv3 import DeepLabV3
from torchvision_wj.models.segwithbox.enet import ENet


class BoostedSegmentationModel(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, softmax: bool, channels_in: int = 16):
        super(BoostedSegmentationModel, self).__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.softmax = softmax

        # Initialize both models
        self.enet = ENet(in_dim, out_dim, softmax, channels_in)
        self.deeplabv3 = DeepLabV3(in_dim, out_dim, softmax, channels_in)

        # Learnable fusion weights
        self.weight_enet = nn.Parameter(torch.tensor(0.5))
        self.weight_deeplab = nn.Parameter(torch.tensor(0.5))

        # Alternative: More sophisticated fusion
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(out_dim * 2, out_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_dim, out_dim, kernel_size=1),
        )

        # Attention mechanism for better fusion
        self.attention = nn.Sequential(
            nn.Conv2d(out_dim * 2, out_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_dim, 2, kernel_size=1),  # 2 channels for attention weights
            nn.Softmax(dim=1),
        )

        # Final activation
        if softmax:
            self.pred_func = nn.Softmax(dim=1)
        else:
            self.pred_func = nn.Sigmoid()

        print(f"Initialized {self.__class__.__name__} successfully")

    def forward(self, x):
        # Get predictions from both models
        enet_output = self.enet(x)[0]  # ENet returns [output]
        deeplab_output = self.deeplabv3(x)[0]  # DeepLabV3 returns [output]

        # Method 1: Simple weighted average
        # output = (self.weight_enet * enet_output + self.weight_deeplab * deeplab_output)

        # Method 2: Concatenation + convolution fusion
        # fused = torch.cat([enet_output, deeplab_output], dim=1)
        # output = self.fusion_conv(fused)

        # Method 3: Attention-based fusion (Recommended)
        output = self.attention_fusion(enet_output, deeplab_output)

        # Apply final activation
        output = self.pred_func(output)

        return [output]

    def attention_fusion(self, enet_out, deeplab_out):
        """Attention-based fusion of both model outputs"""
        # Concatenate features
        concat_features = torch.cat([enet_out, deeplab_out], dim=1)

        # Generate attention weights
        attention_weights = self.attention(concat_features)  # [B, 2, H, W]

        # Apply attention weights
        enet_att = enet_out * attention_weights[:, 0:1, :, :]
        deeplab_att = deeplab_out * attention_weights[:, 1:2, :, :]

        # Fuse attended features
        fused_output = enet_att + deeplab_att

        return fused_output


class MultiScaleBoostedModel(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, softmax: bool, channels_in: int = 16):
        super(MultiScaleBoostedModel, self).__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.softmax = softmax

        # Both models
        self.enet = ENet(in_dim, out_dim, softmax, channels_in)
        self.deeplabv3 = DeepLabV3(in_dim, out_dim, softmax, channels_in)

        # Multi-scale feature fusion
        self.fusion_blocks = nn.ModuleList(
            [
                # High-level feature fusion
                nn.Sequential(
                    nn.Conv2d(out_dim * 2, out_dim, kernel_size=3, padding=1),
                    nn.BatchNorm2d(out_dim),
                    nn.ReLU(inplace=True),
                ),
                # Mid-level feature fusion
                nn.Sequential(
                    nn.Conv2d(out_dim * 2, out_dim, kernel_size=3, padding=1),
                    nn.BatchNorm2d(out_dim),
                    nn.ReLU(inplace=True),
                ),
            ]
        )

        # Final classifier
        self.final_conv = nn.Conv2d(out_dim, out_dim, kernel_size=1)

        if softmax:
            self.pred_func = nn.Softmax(dim=1)
        else:
            self.pred_func = nn.Sigmoid()

    def forward(self, x):
        # Get outputs from both models
        enet_out = self.enet(x)[0]
        deeplab_out = self.deeplabv3(x)[0]

        # Multi-scale fusion
        fused = self.multi_scale_fusion(enet_out, deeplab_out)

        # Final prediction
        output = self.final_conv(fused)
        output = self.pred_func(output)

        return [output]

    def multi_scale_fusion(self, enet_out, deeplab_out):
        """Multi-scale feature fusion"""
        # Original resolution fusion
        fused_high = self.fusion_blocks[0](torch.cat([enet_out, deeplab_out], dim=1))

        # Lower resolution fusion (after pooling)
        enet_low = F.avg_pool2d(enet_out, kernel_size=2, stride=2)
        deeplab_low = F.avg_pool2d(deeplab_out, kernel_size=2, stride=2)

        fused_low = self.fusion_blocks[1](torch.cat([enet_low, deeplab_low], dim=1))

        # Upsample and combine
        fused_low_up = F.interpolate(
            fused_low, size=enet_out.shape[2:], mode="bilinear", align_corners=True
        )

        # Final fusion
        final_fused = fused_high + fused_low_up

        return final_fused


class ResidualBoostedModel(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, softmax: bool, channels_in: int = 16):
        super(ResidualBoostedModel, self).__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.softmax = softmax

        # Base model (ENet)
        self.base_model = ENet(in_dim, out_dim, softmax, channels_in)

        # Boosting model (DeepLabV3) - acts as residual correction
        self.booster_model = DeepLabV3(in_dim, out_dim, softmax, channels_in)

        # Residual fusion
        self.residual_fusion = nn.Sequential(
            nn.Conv2d(out_dim * 2, out_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_dim, out_dim, kernel_size=1),
        )

        # Learnable scaling factor for residual
        self.residual_scale = nn.Parameter(torch.tensor(0.1))

        if softmax:
            self.pred_func = nn.Softmax(dim=1)
        else:
            self.pred_func = nn.Sigmoid()

    def forward(self, x):
        # Base prediction
        base_output = self.base_model(x)[0]

        # Booster prediction (residual)
        booster_output = self.booster_model(x)[0]

        # Residual fusion
        residual_fused = self.residual_fusion(
            torch.cat([base_output, booster_output], dim=1)
        )

        # Add residual to base prediction
        final_output = base_output + self.residual_scale * residual_fused

        # Apply final activation
        final_output = self.pred_func(final_output)

        return [final_output]


class EnsembleBoostedModel(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, softmax: bool, channels_in: int = 16):
        super(EnsembleBoostedModel, self).__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.softmax = softmax

        # Multiple instances for ensemble
        self.enet_models = nn.ModuleList(
            [ENet(in_dim, out_dim, softmax, channels_in) for _ in range(2)]
        )

        self.deeplab_models = nn.ModuleList(
            [DeepLabV3(in_dim, out_dim, softmax, channels_in) for _ in range(2)]
        )

        # Ensemble fusion
        self.ensemble_fusion = nn.Sequential(
            nn.Conv2d(out_dim * 4, out_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_dim, out_dim, kernel_size=1),
        )

        if softmax:
            self.pred_func = nn.Softmax(dim=1)
        else:
            self.pred_func = nn.Sigmoid()

    def forward(self, x):
        # Get predictions from all models
        enet_outputs = [model(x)[0] for model in self.enet_models]
        deeplab_outputs = [model(x)[0] for model in self.deeplab_models]

        # Concatenate all predictions
        all_outputs = torch.cat(enet_outputs + deeplab_outputs, dim=1)

        # Fusion
        fused_output = self.ensemble_fusion(all_outputs)
        final_output = self.pred_func(fused_output)

        return [final_output]


# Usage example:
def create_boosted_model(
    in_dim=1, out_dim=2, softmax=True, ch_in=16, model_type="attention"
):
    """
    Create a boosted segmentation model

    Args:
        model_type: 'attention', 'multi_scale', 'residual', or 'ensemble'
        in_dim: input channels
        out_dim: output channels (number of classes)
        softmax: whether to use softmax activation
    """
    if model_type == "attention":
        return BoostedSegmentationModel(in_dim, out_dim, softmax, ch_in)
    elif model_type == "multi_scale":
        return MultiScaleBoostedModel(in_dim, out_dim, softmax, ch_in)
    elif model_type == "residual":
        return ResidualBoostedModel(in_dim, out_dim, softmax, ch_in)
    elif model_type == "ensemble":
        return EnsembleBoostedModel(in_dim, out_dim, softmax, ch_in)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


# Test the boosted model
if __name__ == "__main__":
    # Create boosted model
    model = create_boosted_model("attention", in_dim=1, out_dim=2, softmax=True)

    # Test input
    input_tensor = torch.randn(2, 1, 256, 256)  # batch_size=2, channels=1, H=256, W=256
    output = model(input_tensor)

    print(f"Input shape: {input_tensor.shape}")
    print(f"Output shape: {output[0].shape}")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters()):,}")
