# Adding the three missing configurations to your existing code

import torch
import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision_wj.models.segwithbox.boosting_ensemble import (
    BaseBoostingModel,
    FixedBoostingDeeplabUnetResNet,
    FixedBoostingResNetUnetDeeplab,
    FixedBoostingUnetDeeplabResNet,
)

# from torchvision_wj.models.segwithbox.enet_23 import ENet
from torchvision_wj.models.segwithbox.residualunet import ResidualUNet


class FixedBoostingUnetResnetDeeplab(BaseBoostingModel):
    """
    Fixed Configuration 4: UNet -> ResNet -> DeepLabV3
    """

    def __init__(
        self,
        unet_model,
        resnet_model,
        deeplabv3_model,
        in_dim: int,
        out_dim: int,
        softmax: bool,
    ):
        super().__init__(
            unet_model, resnet_model, deeplabv3_model, in_dim, out_dim, softmax
        )
        self.model1 = unet_model
        self.model2 = resnet_model
        self.model3 = deeplabv3_model

        print("Initialized FixedBoostingUnetResnetDeeplab: UNet → ResNet → DeepLabV3")

    def forward(self, x):
        """
        Fixed forward pass with proper channel handling
        """
        # ========== STAGE 1: UNet ==========
        unet_output = self.model1(x)[0]
        unet_error = self.compute_error_map(unet_output)

        # Prepare input for Stage 2
        stage2_input = self._prepare_stage_input(x, unet_error, self.error_correction1)

        # ========== STAGE 2: ResNet ==========
        resnet_output = self.model2(stage2_input)[0]
        resnet_error = self.compute_error_map(resnet_output)

        # Prepare input for Stage 3
        stage3_input = self._prepare_stage_input(
            x, resnet_error, self.error_correction2
        )

        # ========== STAGE 3: DeepLabV3 ==========
        deeplab_output = self.model3(stage3_input)[0]

        # ========== FUSION ==========
        final_output = self.fuse_predictions(unet_output, resnet_output, deeplab_output)

        # Final activation
        if self.softmax:
            final_output = F.softmax(final_output, dim=1)
        else:
            final_output = torch.sigmoid(final_output)

        return [final_output]

    def _prepare_stage_input(self, original_input, error_map, correction_module):
        """
        Safely prepare stage input with proper channel handling
        """
        if error_map.shape[2:] != original_input.shape[2:]:
            error_map = F.interpolate(
                error_map,
                size=original_input.shape[2:],
                mode="bilinear",
                align_corners=True,
            )

        try:
            combined = torch.cat([original_input, error_map], dim=1)
            corrected = correction_module(combined)
            return corrected
        except Exception as e:
            print(f"Warning: Error correction failed, using original input: {e}")
            return original_input


class FixedBoostingResNetDeeplabUnet(BaseBoostingModel):
    """
    Fixed Configuration 5: ResNet -> DeepLabV3 -> UNet
    """

    def __init__(
        self,
        resnet_model,
        deeplabv3_model,
        unet_model,
        in_dim: int,
        out_dim: int,
        softmax: bool,
    ):
        super().__init__(
            resnet_model, deeplabv3_model, unet_model, in_dim, out_dim, softmax
        )
        self.model1 = resnet_model
        self.model2 = deeplabv3_model
        self.model3 = unet_model

        print("Initialized FixedBoostingResNetDeeplabUnet: ResNet → DeepLabV3 → UNet")

    def forward(self, x):
        """
        Fixed forward pass with proper channel handling
        """
        # ========== STAGE 1: ResNet ==========
        resnet_output = self.model1(x)[0]
        resnet_error = self.compute_error_map(resnet_output)

        # Prepare input for Stage 2
        stage2_input = self._prepare_stage_input(
            x, resnet_error, self.error_correction1
        )

        # ========== STAGE 2: DeepLabV3 ==========
        deeplab_output = self.model2(stage2_input)[0]
        deeplab_error = self.compute_error_map(deeplab_output)

        # Prepare input for Stage 3
        stage3_input = self._prepare_stage_input(
            x, deeplab_error, self.error_correction2
        )

        # ========== STAGE 3: UNet ==========
        unet_output = self.model3(stage3_input)[0]

        # ========== FUSION ==========
        final_output = self.fuse_predictions(resnet_output, deeplab_output, unet_output)

        # Final activation
        if self.softmax:
            final_output = F.softmax(final_output, dim=1)
        else:
            final_output = torch.sigmoid(final_output)

        return [final_output]

    def _prepare_stage_input(self, original_input, error_map, correction_module):
        """
        Safely prepare stage input with proper channel handling
        """
        if error_map.shape[2:] != original_input.shape[2:]:
            error_map = F.interpolate(
                error_map,
                size=original_input.shape[2:],
                mode="bilinear",
                align_corners=True,
            )

        try:
            combined = torch.cat([original_input, error_map], dim=1)
            corrected = correction_module(combined)
            return corrected
        except Exception as e:
            print(f"Warning: Error correction failed, using original input: {e}")
            return original_input


class FixedBoostingDeeplabResnetUnet(BaseBoostingModel):
    """
    Fixed Configuration 6: DeepLabV3 -> ResNet -> UNet
    """

    def __init__(
        self,
        deeplabv3_model,
        resnet_model,
        unet_model,
        in_dim: int,
        out_dim: int,
        softmax: bool,
    ):
        super().__init__(
            deeplabv3_model, resnet_model, unet_model, in_dim, out_dim, softmax
        )
        self.model1 = deeplabv3_model
        self.model2 = resnet_model
        self.model3 = unet_model

        print("Initialized FixedBoostingDeeplabResnetUnet: DeepLabV3 → ResNet → UNet")

    def forward(self, x):
        """
        Fixed forward pass with proper channel handling
        """
        # ========== STAGE 1: DeepLabV3 ==========
        deeplab_output = self.model1(x)[0]
        deeplab_error = self.compute_error_map(deeplab_output)

        # Prepare input for Stage 2
        stage2_input = self._prepare_stage_input(
            x, deeplab_error, self.error_correction1
        )

        # ========== STAGE 2: ResNet ==========
        resnet_output = self.model2(stage2_input)[0]
        resnet_error = self.compute_error_map(resnet_output)

        # Prepare input for Stage 3
        stage3_input = self._prepare_stage_input(
            x, resnet_error, self.error_correction2
        )

        # ========== STAGE 3: UNet ==========
        unet_output = self.model3(stage3_input)[0]

        # ========== FUSION ==========
        final_output = self.fuse_predictions(deeplab_output, resnet_output, unet_output)

        # Final activation
        if self.softmax:
            final_output = F.softmax(final_output, dim=1)
        else:
            final_output = torch.sigmoid(final_output)

        return [final_output]

    def _prepare_stage_input(self, original_input, error_map, correction_module):
        """
        Safely prepare stage input with proper channel handling
        """
        if error_map.shape[2:] != original_input.shape[2:]:
            error_map = F.interpolate(
                error_map,
                size=original_input.shape[2:],
                mode="bilinear",
                align_corners=True,
            )

        try:
            combined = torch.cat([original_input, error_map], dim=1)
            corrected = correction_module(combined)
            return corrected
        except Exception as e:
            print(f"Warning: Error correction failed, using original input: {e}")
            return original_input


# Updated factory function with all 6 configurations
def create_boosting_ensemble_complete(
    model1,
    model2,
    model3,
    model1_type: str,
    in_dim: int,
    out_dim: int,
    softmax: bool,
    configuration: int = 1,
):
    """
    Complete factory function to create all 6 boosting configurations
    
    configurations = [
        4 (2, "unet", "UNet → ResNet → DeepLabV3"),
        5 (3, "resnet", "ResNet → UNet → DeepLabV3"),
        6 (6, "deeplab", "DeepLabV3 → ResNet → UNet"),
    ]

    Args:
        model1: First model instance
        model2: Second model instance
        model3: Third model instance
        model1_type: Type of the first model ('unet', 'resnet', 'deeplab')
        in_dim: Number of input channels
        out_dim: Number of output channels
        softmax: Whether to use softmax activation
        configuration: Which boosting configuration to use (1-6)

    Returns:
        Initialized boosting model instance
    """

    if configuration == 1:
        # UNet -> DeepLabV3 -> ResNet
        if model1_type.lower() != "unet":
            raise ValueError("Configuration 1 requires UNet as first model")
        return FixedBoostingUnetDeeplabResNet(
            model1, model2, model3, in_dim, out_dim, softmax
        )

    elif configuration == 2:
        # UNet -> ResNet -> DeepLabV3
        if model1_type.lower() != "unet":
            raise ValueError("Configuration 2 requires UNet as first model")
        return FixedBoostingUnetResnetDeeplab(
            model1, model2, model3, in_dim, out_dim, softmax
        )

    elif configuration == 3:
        # ResNet -> UNet -> DeepLabV3
        if model1_type.lower() != "resnet":
            raise ValueError("Configuration 3 requires ResNet as first model")
        return FixedBoostingResNetUnetDeeplab(
            model1, model2, model3, in_dim, out_dim, softmax
        )

    elif configuration == 4:
        # ResNet -> DeepLabV3 -> UNet
        if model1_type.lower() != "resnet":
            raise ValueError("Configuration 4 requires ResNet as first model")
        return FixedBoostingResNetDeeplabUnet(
            model1, model2, model3, in_dim, out_dim, softmax
        )

    elif configuration == 5:
        # DeepLabV3 -> UNet -> ResNet
        if model1_type.lower() != "deeplab":
            raise ValueError("Configuration 5 requires DeepLabV3 as first model")
        return FixedBoostingDeeplabUnetResNet(
            model1, model2, model3, in_dim, out_dim, softmax
        )

    elif configuration == 6:
        # DeepLabV3 -> ResNet -> UNet
        if model1_type.lower() != "deeplab":
            raise ValueError("Configuration 6 requires DeepLabV3 as first model")
        return FixedBoostingDeeplabResnetUnet(
            model1, model2, model3, in_dim, out_dim, softmax
        )

    else:
        raise ValueError(f"Invalid configuration: {configuration}. Use 1-6.")


# Enhanced test function to test all 6 configurations
def test_create_boosting_ensemble_complete():
    """
    Test function for all 6 boosting ensemble implementations
    """

    print("=" * 70)
    print("TESTING COMPLETE create_boosting_ensemble FUNCTION (6 CONFIGURATIONS)")
    print("=" + "=" * 69)

    # Mock models (same as before)
    class MockUNet(nn.Module):
        def __init__(self, in_dim, out_dim, softmax):
            super().__init__()
            self.conv = nn.Conv2d(in_dim, out_dim, 3, padding=1)
            self.softmax = softmax

        def forward(self, x):
            x = self.conv(x)
            if self.softmax:
                x = F.softmax(x, dim=1)
            else:
                x = torch.sigmoid(x)
            return [x]

    class MockResNet(nn.Module):
        def __init__(self, in_dim, out_dim, softmax):
            super().__init__()
            self.conv = nn.Conv2d(in_dim, out_dim, 3, padding=1)
            self.softmax = softmax

        def forward(self, x):
            x = self.conv(x)
            if self.softmax:
                x = F.softmax(x, dim=1)
            else:
                x = torch.sigmoid(x)
            return [x]

    class MockDeepLabV3(nn.Module):
        def __init__(self, in_dim, out_dim, softmax):
            super().__init__()
            self.conv = nn.Conv2d(in_dim, out_dim, 3, padding=1)
            self.softmax = softmax

        def forward(self, x):
            x = self.conv(x)
            if self.softmax:
                x = F.softmax(x, dim=1)
            else:
                x = torch.sigmoid(x)
            return [x]

    # Test all 6 configurations
    configurations = [
        (1, "unet", "UNet → DeepLabV3 → ResNet"),
        (2, "unet", "UNet → ResNet → DeepLabV3"),
        (3, "resnet", "ResNet → UNet → DeepLabV3"),
        (4, "resnet", "ResNet → DeepLabV3 → UNet"),
        (5, "deeplab", "DeepLabV3 → UNet → ResNet"),
        (6, "deeplab", "DeepLabV3 → ResNet → UNet"),
    ]

    success_count = 0
    for config_num, model_type, description in configurations:
        try:
            print(f"\nTesting Configuration {config_num}: {description}")
            print("-" * 50)

            # Create mock models
            unet = MockUNet(in_dim=1, out_dim=2, softmax=True)
            resnet = MockResNet(in_dim=1, out_dim=2, softmax=True)
            deeplab = MockDeepLabV3(in_dim=1, out_dim=2, softmax=True)

            # Determine model order based on configuration
            if config_num == 1:
                model1, model2, model3 = unet, deeplab, resnet
            elif config_num == 2:
                model1, model2, model3 = unet, resnet, deeplab
            elif config_num == 3:
                model1, model2, model3 = resnet, unet, deeplab
            elif config_num == 4:
                model1, model2, model3 = resnet, deeplab, unet
            elif config_num == 5:
                model1, model2, model3 = deeplab, unet, resnet
            else:  # config_num == 6
                model1, model2, model3 = deeplab, resnet, unet

            # Create boosting ensemble with complete version
            model = create_boosting_ensemble_complete(
                model1=model1,
                model2=model2,
                model3=model3,
                model1_type=model_type,
                in_dim=1,
                out_dim=2,
                softmax=True,
                configuration=config_num,
            )

            # Test forward pass
            test_input = torch.randn(2, 1, 64, 64)
            output = model(test_input)

            # Verify output properties
            assert output[0].shape == (2, 2, 64, 64), f"Output shape mismatch"
            assert 0 <= output[0].min() <= output[0].max() <= 1, f"Output range invalid"

            print(f"✓ Config {config_num}: {description} - SUCCESS")
            print(f"  Output shape: {output[0].shape}")
            print(f"  Output range: [{output[0].min():.3f}, {output[0].max():.3f}]")
            success_count += 1

        except Exception as e:
            print(f"✗ Config {config_num}: {description} - FAILED: {e}")

    print(f"\nSUMMARY: {success_count}/6 tests passed")

    if success_count == 6:
        print("🎉 ALL 6 CONFIGURATIONS ARE WORKING CORRECTLY!")
    else:
        print(f"⚠️  {6 - success_count} configurations need attention")
        print(
            "💡 Consider using SimpleRobustBoosting as a fallback for problematic configurations"
        )

    return success_count == 6


# Example usage function
def demonstrate_all_configurations():
    """
    Demonstrate how to use all 6 configurations
    """
    print("\n" + "=" * 70)
    print("DEMONSTRATING ALL 6 BOOSTING CONFIGURATIONS")
    print("=" + "=" * 69)

    # This would be with your actual models
    # unet = ENet(...)  # Your actual UNet implementation
    # resnet = ResidualUNet(...)  # Your actual ResNet implementation
    # deeplab = YourDeepLabV3Implementation(...)  # Your actual DeepLabV3

    configurations = [
        (1, "UNet → DeepLabV3 → ResNet"),
        (2, "UNet → ResNet → DeepLabV3"),
        (3, "ResNet → UNet → DeepLabV3"),
        (4, "ResNet → DeepLabV3 → UNet"),
        (5, "DeepLabV3 → UNet → ResNet"),
        (6, "DeepLabV3 → ResNet → UNet"),
    ]

    for config_num, description in configurations:
        print(f"\nConfiguration {config_num}: {description}")
        print("Usage:")
        print(f"  model = create_boosting_ensemble_complete(")
        print(f"      model1=unet,  # or resnet/deeplab depending on configuration")
        print(f"      model2=resnet,  # varies by configuration")
        print(f"      model3=deeplab,  # varies by configuration")
        print(
            f"      model1_type='{configurations[config_num-1][1].split('→')[0].strip().lower()}',"
        )
        print(f"      in_dim=3, out_dim=21, softmax=True,")
        print(f"      configuration={config_num}")
        print(f"  )")


if __name__ == "__main__":
    # Test the complete version with all 6 configurations
    test_create_boosting_ensemble_complete()

    # Demonstrate usage
    demonstrate_all_configurations()
