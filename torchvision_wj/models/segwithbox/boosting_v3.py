import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision_wj.models.segwithbox.DeepLabv3 import DeepLabV3
from torchvision_wj.models.segwithbox.enet_23 import ENet
from torchvision_wj.models.segwithbox.residualunet import ResidualUNet


class SequentialBoostingEnsemble(nn.Module):
    def __init__(
        self,
        enet_model,
        resunet_model,
        deeplabv3_model,
        in_dim: int,
        out_dim: int,
        softmax: bool,
    ):
        super().__init__()
        self.enet = enet_model
        self.resunet = resunet_model
        self.deeplabv3 = deeplabv3_model
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.softmax = softmax

        # Error correction modules for sequential boosting
        self.error_correction1 = nn.Sequential(
            nn.Conv2d(out_dim + in_dim, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, in_dim, kernel_size=1),
        )

        self.error_correction2 = nn.Sequential(
            nn.Conv2d(out_dim + in_dim, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, in_dim, kernel_size=1),
        )

        # Confidence-based fusion weights
        self.confidence_weights = nn.Parameter(torch.ones(3))

        # Final refinement
        self.final_refinement = nn.Sequential(
            nn.Conv2d(out_dim * 3, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, out_dim, kernel_size=1),
        )

        # Attention mechanism for feature selection
        self.feature_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_dim * 3, out_dim, kernel_size=1),
            nn.Sigmoid(),
        )

        print(
            f"Initialized SequentialBoostingEnsemble with {in_dim} input channels and {out_dim} output channels"
        )

    def compute_confidence(self, predictions, targets=None):
        """Compute confidence scores for predictions"""
        if self.softmax:
            # Use entropy as confidence measure (lower entropy = higher confidence)
            entropy = -torch.sum(predictions * torch.log(predictions + 1e-8), dim=1)
            confidence = 1.0 / (1.0 + entropy.unsqueeze(1))
        else:
            # For sigmoid, use maximum probability
            confidence = torch.max(predictions, 1 - predictions, dim=1)[0].unsqueeze(1)
        return confidence

    def compute_error_map(self, predictions):
        """Compute error maps based on prediction uncertainty"""
        if self.softmax:
            # Use prediction entropy as error measure
            max_prob, _ = torch.max(predictions, dim=1, keepdim=True)
            error_map = 1.0 - max_prob
        else:
            # For binary case, use distance from 0.5
            error_map = 2.0 * torch.abs(predictions - 0.5)
        return error_map

    def forward(self, x):
        batch_size = x.shape[0]

        # ========== STAGE 1: ENet (Fast, efficient) ==========
        enet_output = self.enet(x)[0]  # [B, C, H, W]
        enet_confidence = self.compute_confidence(enet_output)
        enet_error = self.compute_error_map(enet_output)

        # Prepare input for next stage: original image + error guidance
        stage2_input = torch.cat([x, enet_error], dim=1)
        if hasattr(self, "error_correction1"):
            stage2_input = self.error_correction1(stage2_input)
        else:
            # Simple concatenation if no correction module
            stage2_input = x + enet_error * 0.1  # Weighted error addition

        # ========== STAGE 2: ResidualUNet (Good balance) ==========
        resunet_output = self.resunet(stage2_input)[0]
        resunet_confidence = self.compute_confidence(resunet_output)
        resunet_error = self.compute_error_map(resunet_output)

        # Prepare input for next stage
        stage3_input = torch.cat([x, resunet_error], dim=1)
        if hasattr(self, "error_correction2"):
            stage3_input = self.error_correction2(stage3_input)
        else:
            stage3_input = x + resunet_error * 0.1

        # ========== STAGE 3: DeepLabV3 (Strong context) ==========
        deeplab_output = self.deeplabv3(stage3_input)[0]
        deeplab_confidence = self.compute_confidence(deeplab_output)

        # ========== CONFIDENCE-BASED FUSION ==========
        # Normalize confidence scores
        confidences = torch.stack(
            [
                enet_confidence.mean(),
                resunet_confidence.mean(),
                deeplab_confidence.mean(),
            ]
        )
        normalized_weights = F.softmax(confidences, dim=0)

        # Learnable weights combined with confidence
        final_weights = F.softmax(self.confidence_weights * normalized_weights, dim=0)

        # Weighted fusion
        weighted_fusion = (
            final_weights[0] * enet_output
            + final_weights[1] * resunet_output
            + final_weights[2] * deeplab_output
        )

        # ========== REFINEMENT FUSION ==========
        # Alternative: Feature concatenation + refinement
        concatenated_features = torch.cat(
            [enet_output, resunet_output, deeplab_output], dim=1
        )
        refined_output = self.final_refinement(concatenated_features)

        # ========== ATTENTION-BASED SELECTION ==========
        # Use attention to select best features
        attention_weights = self.feature_attention(concatenated_features)
        attended_output = concatenated_features * attention_weights
        attended_output = (
            attended_output.sum(dim=1, keepdim=True)
            if self.out_dim == 1
            else attended_output
        )

        # ========== FINAL COMBINATION ==========
        # Combine weighted fusion with refined output
        if hasattr(self, "final_refinement"):
            final_output = 0.7 * refined_output + 0.3 * weighted_fusion
        else:
            final_output = weighted_fusion

        # Apply final activation
        if self.softmax:
            final_output = F.softmax(final_output, dim=1)
        else:
            final_output = torch.sigmoid(final_output)

        return [final_output]


class ProgressiveBoostingEnsemble(nn.Module):
    """Alternative: Progressive boosting where each model refines the previous"""

    def __init__(
        self,
        enet_model,
        resunet_model,
        deeplabv3_model,
        in_dim: int,
        out_dim: int,
        softmax: bool,
    ):
        super().__init__()
        self.enet = enet_model
        self.resunet = resunet_model
        self.deeplabv3 = deeplabv3_model
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.softmax = softmax

        # Progressive refinement modules
        self.refinement1 = nn.Sequential(
            nn.Conv2d(out_dim + in_dim, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, out_dim, kernel_size=1),
        )

        self.refinement2 = nn.Sequential(
            nn.Conv2d(out_dim * 2 + in_dim, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, out_dim, kernel_size=1),
        )

        # Residual connections
        self.residual_conv1 = nn.Conv2d(out_dim, out_dim, kernel_size=1)
        self.residual_conv2 = nn.Conv2d(out_dim, out_dim, kernel_size=1)

    def forward(self, x):
        # Stage 1: ENet - Fast initial segmentation
        enet_out = self.enet(x)[0]

        # Refine ENet output with original image
        stage1_refined = self.refinement1(torch.cat([enet_out, x], dim=1))
        stage1_final = enet_out + self.residual_conv1(stage1_refined)

        if self.softmax:
            stage1_final = F.softmax(stage1_final, dim=1)
        else:
            stage1_final = torch.sigmoid(stage1_final)

        # Stage 2: ResidualUNet - Refine based on stage1 output
        resunet_out = self.resunet(x)[0]

        # Combine stage1 output with ResidualUNet output and original image
        stage2_input = torch.cat([stage1_final, resunet_out, x], dim=1)
        stage2_refined = self.refinement2(stage2_input)
        stage2_final = resunet_out + self.residual_conv2(stage2_refined)

        if self.softmax:
            stage2_final = F.softmax(stage2_final, dim=1)
        else:
            stage2_final = torch.sigmoid(stage2_final)

        # Stage 3: DeepLabV3 - Final refinement with global context
        deeplab_out = self.deeplabv3(x)[0]

        # Adaptive fusion of all stages
        final_output = stage1_final * 0.2 + stage2_final * 0.3 + deeplab_out * 0.5

        if self.softmax:
            final_output = F.softmax(final_output, dim=1)
        else:
            final_output = torch.sigmoid(final_output)

        return [final_output]


class AdaptiveBoostingEnsemble(nn.Module):
    """Adaptive boosting that learns which model to trust in different regions"""

    def __init__(
        self,
        enet_model,
        resunet_model,
        deeplabv3_model,
        in_dim: int,
        out_dim: int,
        softmax: bool,
    ):
        super().__init__()
        self.enet = enet_model
        self.resunet = resunet_model
        self.deeplabv3 = deeplabv3_model
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.softmax = softmax

        # Spatial attention for model selection
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(out_dim * 3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 3, kernel_size=1),  # 3 attention maps for 3 models
            nn.Softmax(dim=1),
        )

        # Confidence estimation
        self.confidence_net = nn.Sequential(
            nn.Conv2d(out_dim, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # Get predictions from all models
        enet_out = self.enet(x)[0]
        resunet_out = self.resunet(x)[0]
        deeplab_out = self.deeplabv3(x)[0]

        # Concatenate all predictions
        all_outputs = torch.cat([enet_out, resunet_out, deeplab_out], dim=1)

        # Generate spatial attention weights
        attention_weights = self.spatial_attention(all_outputs)

        # Apply attention weights
        attended_enet = enet_out * attention_weights[:, 0:1, :, :]
        attended_resunet = resunet_out * attention_weights[:, 1:2, :, :]
        attended_deeplab = deeplab_out * attention_weights[:, 2:3, :, :]

        # Combine attended features
        final_output = attended_enet + attended_resunet + attended_deeplab

        # Apply final activation
        if self.softmax:
            final_output = F.softmax(final_output, dim=1)
        else:
            final_output = torch.sigmoid(final_output)

        return [final_output]


# Factory function to create boosting models
def create_sequential_boosting_v3(
    in_dim: int,
    out_dim: int,
    softmax: bool,
    ch_in=32,
    model_type: str = "sequential",
):
    """
    Create a sequential boosting ensemble model

    Args:
        enet_model: ENet model instance
        resunet_model: ResidualUNet model instance
        deeplabv3_model: DeepLabV3 model instance
        in_dim: Input channels
        out_dim: Output channels
        softmax: Whether to use softmax activation
        model_type: 'sequential', 'progressive', or 'adaptive'
    """
    enet_model = ENet(in_dim, out_dim, softmax, ch_in)
    resunet_model = ResidualUNet(in_dim, out_dim, softmax, ch_in)
    deeplabv3_model = DeepLabV3(in_dim, out_dim, softmax, ch_in)
    if model_type == "sequential":
        return SequentialBoostingEnsemble(
            enet_model, resunet_model, deeplabv3_model, in_dim, out_dim, softmax
        )
    elif model_type == "progressive":
        return ProgressiveBoostingEnsemble(
            enet_model, resunet_model, deeplabv3_model, in_dim, out_dim, softmax
        )
    elif model_type == "adaptive":
        return AdaptiveBoostingEnsemble(
            enet_model, resunet_model, deeplabv3_model, in_dim, out_dim, softmax
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")


# Example usage:
if __name__ == "__main__":
    # Assuming you have initialized your models elsewhere
    enet = ENet(in_dim=1, out_dim=2, softmax=True)
    resunet = ResidualUNet(in_dim=1, out_dim=2, softmax=True)
    deeplabv3 = DeepLabV3(in_dim=1, out_dim=2, softmax=True)

    # Create sequential boosting model
    boosting_model = create_sequential_boosting(
        enet_model=enet,
        resunet_model=resunet,
        deeplabv3_model=deeplabv3,
        in_dim=1,
        out_dim=2,
        softmax=True,
        model_type="sequential",
    )

    # Test the model
    test_input = torch.randn(2, 1, 256, 256)
    output = boosting_model(test_input)
    print(f"Input shape: {test_input.shape}")
    print(f"Output shape: {output[0].shape}")
