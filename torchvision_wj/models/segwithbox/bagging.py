import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Optional, Union


class AdvancedBaggingEnsemble(nn.Module):
    """
    Advanced Bagging Ensemble with weighted voting, uncertainty estimation,
    and adaptive model selection
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        softmax: bool = True,
        device: torch.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        ),
    ):
        super().__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.softmax = softmax
        self.device = device

        # Initialize models with different architectures for diversity
        self.unet = SimpleUNet(in_dim, out_dim, softmax=False)
        self.resnet = SimpleResNet(in_dim, out_dim, softmax=False)
        self.deeplab = SimpleDeepLabV3(in_dim, out_dim, softmax=False)

        # Learnable weights for each model (initialized based on typical performance)
        self.model_weights = nn.Parameter(
            torch.tensor([0.34, 0.33, 0.33])
        )  # UNet, ResNet, DeepLab

        # Uncertainty-based weighting
        self.uncertainty_weight = nn.Parameter(torch.tensor(0.5))

        # Adaptive dropout rates for bagging
        self.dropout_rates = nn.Parameter(torch.tensor([0.1, 0.15, 0.2]))

        # Confidence calibration
        self.temperature = nn.Parameter(torch.tensor(1.0))

        # Feature alignment (if needed for later fusion)
        self.feature_align = nn.ModuleDict(
            {
                "unet": nn.Conv2d(32, 64, 1),
                "resnet": nn.Conv2d(32, 64, 1),
                "deeplab": nn.Conv2d(16, 64, 1),  # deeplab final channels = 64//4 = 16
            }
        )

        self._initialize_models()

    def _initialize_models(self):
        """Initialize models with different random seeds for diversity"""

        def init_weights(m):
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        # Different initialization for each model
        torch.manual_seed(42)
        self.unet.apply(init_weights)

        torch.manual_seed(123)
        self.resnet.apply(init_weights)

        torch.manual_seed(456)
        self.deeplab.apply(init_weights)

    def forward(
        self, x: torch.Tensor, return_individual: bool = False
    ) -> Union[torch.Tensor, tuple]:
        """
        Forward pass with advanced ensemble methods

        Args:
            x: Input tensor
            return_individual: Whether to return individual model outputs

        Returns:
            Ensemble prediction and optionally individual outputs
        """
        batch_size = x.shape[0]

        # Get predictions from all models with dropout for bagging
        individual_outputs = self._get_individual_predictions(x)

        # Apply temperature scaling for calibration
        individual_outputs = self._apply_temperature_scaling(individual_outputs)

        # Calculate model uncertainties
        uncertainties = self._calculate_uncertainties(individual_outputs)

        # Compute adaptive weights based on uncertainty
        adaptive_weights = self._compute_adaptive_weights(uncertainties)

        # Combine predictions using weighted average
        ensemble_output = self._weighted_ensemble(individual_outputs, adaptive_weights)

        # Apply final activation
        if self.softmax:
            ensemble_output = F.softmax(ensemble_output, dim=1)
        else:
            ensemble_output = torch.sigmoid(ensemble_output)

        if return_individual:
            return ensemble_output, individual_outputs, uncertainties
        return ensemble_output

    def _get_individual_predictions(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Get predictions from individual models with bagging dropout"""
        outputs = {}

        # UNet prediction with dropout
        if self.training:
            unet_out = self.unet(x)[0]
            # Apply channel dropout for bagging
            mask = torch.bernoulli(
                torch.ones_like(unet_out) * (1 - self.dropout_rates[0])
            )
            unet_out = unet_out * mask
        else:
            unet_out = self.unet(x)[0]
        outputs["unet"] = unet_out

        # ResNet prediction with dropout
        if self.training:
            resnet_out = self.resnet(x)[0]
            mask = torch.bernoulli(
                torch.ones_like(resnet_out) * (1 - self.dropout_rates[1])
            )
            resnet_out = resnet_out * mask
        else:
            resnet_out = self.resnet(x)[0]
        outputs["resnet"] = resnet_out

        # DeepLab prediction with dropout
        if self.training:
            deeplab_out = self.deeplab(x)[0]
            mask = torch.bernoulli(
                torch.ones_like(deeplab_out) * (1 - self.dropout_rates[2])
            )
            deeplab_out = deeplab_out * mask
        else:
            deeplab_out = self.deeplab(x)[0]
        outputs["deeplab"] = deeplab_out

        return outputs

    def _apply_temperature_scaling(
        self, outputs: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Apply temperature scaling for better calibration"""
        scaled_outputs = {}
        for name, output in outputs.items():
            if self.softmax:
                # For multi-class, scale before softmax
                scaled_outputs[name] = output / self.temperature
            else:
                # For binary, scale logits
                scaled_outputs[name] = output / self.temperature
        return scaled_outputs

    def _calculate_uncertainties(
        self, outputs: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Calculate uncertainty estimates for each model"""
        uncertainties = {}

        for name, output in outputs.items():
            if self.softmax:
                # Use entropy as uncertainty measure for multi-class
                probs = F.softmax(output, dim=1)
                entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)
                uncertainties[name] = entropy.unsqueeze(1)
            else:
                # Use variance-like measure for binary
                uncertainty = torch.abs(output - 0.5)  # Distance from decision boundary
                uncertainties[name] = uncertainty

        return uncertainties

    def _compute_adaptive_weights(
        self, uncertainties: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Compute adaptive weights based on uncertainty and learned weights"""
        batch_size = next(iter(uncertainties.values())).shape[0]

        # Base weights (learned)
        base_weights = F.softmax(self.model_weights, dim=0)

        # Uncertainty-based adjustment (lower uncertainty -> higher weight)
        uncertainty_weights = torch.ones(3, device=self.device)
        for i, (name, uncertainty) in enumerate(uncertainties.items()):
            # Average uncertainty over batch and spatial dimensions
            avg_uncertainty = torch.mean(uncertainty)
            uncertainty_weights[i] = 1.0 / (1.0 + avg_uncertainty)

        # Normalize uncertainty weights
        uncertainty_weights = uncertainty_weights / torch.sum(uncertainty_weights)

        # Combine base weights and uncertainty weights
        alpha = torch.sigmoid(self.uncertainty_weight)
        final_weights = alpha * base_weights + (1 - alpha) * uncertainty_weights

        # Expand for batch dimension
        final_weights = final_weights.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        final_weights = final_weights.expand(batch_size, 3, 1, 1, 1)

        return final_weights

    def _weighted_ensemble(
        self, outputs: Dict[str, torch.Tensor], weights: torch.Tensor
    ) -> torch.Tensor:
        """Combine predictions using weighted average"""
        output_list = [outputs["unet"], outputs["resnet"], outputs["deeplab"]]
        stacked_outputs = torch.stack(output_list, dim=1)  # [B, 3, C, H, W]

        # Apply weights and sum
        weighted_outputs = stacked_outputs * weights
        ensemble_output = torch.sum(weighted_outputs, dim=1)

        return ensemble_output

    def predict_with_confidence(self, x: torch.Tensor) -> tuple:
        """
        Get prediction with confidence scores
        Returns: (prediction, confidence_map, uncertainty_map)
        """
        with torch.no_grad():
            ensemble_output, individual_outputs, uncertainties = self.forward(
                x, return_individual=True
            )

            if self.softmax:
                # For multi-class: confidence is max probability
                confidence, _ = torch.max(ensemble_output, dim=1)
                # Overall uncertainty is average of individual uncertainties
                uncertainty_stack = torch.stack(list(uncertainties.values()), dim=1)
                uncertainty_map = torch.mean(uncertainty_stack, dim=1)
            else:
                # For binary: confidence is distance from 0.5
                confidence = torch.abs(ensemble_output - 0.5) * 2
                uncertainty_map = torch.mean(
                    torch.stack(list(uncertainties.values()), dim=1), dim=1
                )

            return (
                ensemble_output,
                confidence.unsqueeze(1),
                uncertainty_map.unsqueeze(1),
            )

    def get_model_performance_metrics(
        self, x: torch.Tensor, y: torch.Tensor
    ) -> Dict[str, float]:
        """
        Evaluate individual model performance on batch
        Useful for adaptive weighting
        """
        metrics = {}

        with torch.no_grad():
            individual_outputs = self._get_individual_predictions(x)

            for name, output in individual_outputs.items():
                if self.softmax:
                    pred = F.softmax(output, dim=1)
                    # Calculate IoU for each model
                    iou = self._calculate_iou(pred, y)
                    metrics[f"{name}_iou"] = iou
                else:
                    pred = torch.sigmoid(output)
                    dice = self._calculate_dice(pred, y)
                    metrics[f"{name}_dice"] = dice

        return metrics

    def _calculate_iou(self, pred: torch.Tensor, target: torch.Tensor) -> float:
        """Calculate mean IoU"""
        # Convert to hard predictions
        pred_mask = torch.argmax(pred, dim=1)
        target_mask = torch.argmax(target, dim=1) if target.dim() > 1 else target

        ious = []
        for class_id in range(self.out_dim):
            pred_class = pred_mask == class_id
            target_class = target_mask == class_id

            intersection = (pred_class & target_class).float().sum()
            union = (pred_class | target_class).float().sum()

            if union > 0:
                ious.append(intersection / union)

        return torch.mean(torch.tensor(ious)).item()

    def _calculate_dice(self, pred: torch.Tensor, target: torch.Tensor) -> float:
        """Calculate Dice coefficient"""
        smooth = 1e-8
        intersection = (pred * target).sum()
        dice = (2.0 * intersection + smooth) / (pred.sum() + target.sum() + smooth)
        return dice.item()


# Enhanced training wrapper for the ensemble
class EnsembleTrainer:
    """Training helper for the ensemble model"""

    def __init__(
        self,
        model: AdvancedBaggingEnsemble,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        device: torch.device,
    ):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device

    def train_step(self, x: torch.Tensor, y: torch.Tensor) -> Dict[str, float]:
        """Single training step"""
        self.model.train()
        self.optimizer.zero_grad()

        # Forward pass
        output = self.model(x)

        # Calculate loss
        if isinstance(output, tuple):
            output = output[0]

        loss = self.criterion(output, y)

        # Backward pass
        loss.backward()
        self.optimizer.step()

        return {"loss": loss.item()}

    def validate(self, val_loader: torch.utils.data.DataLoader) -> Dict[str, float]:
        """Validation step"""
        self.model.eval()
        total_loss = 0
        total_samples = 0

        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(self.device), y.to(self.device)
                output = self.model(x)

                if isinstance(output, tuple):
                    output = output[0]

                loss = self.criterion(output, y)
                total_loss += loss.item() * x.size(0)
                total_samples += x.size(0)

        return {"val_loss": total_loss / total_samples}


# Example usage and testing
if __name__ == "__main__":
    # Configuration - PLEASE UPDATE THESE BASED ON YOUR DATA
    BATCH_SIZE = 4
    IMG_HEIGHT = 256
    IMG_WIDTH = 256
    IN_CHANNELS = 3
    NUM_CLASSES = 2  # Update this!
    IS_BINARY = NUM_CLASSES == 1  # Update this!

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create ensemble
    model = AdvancedBaggingEnsemble(
        in_dim=IN_CHANNELS,
        out_dim=1 if IS_BINARY else NUM_CLASSES,
        softmax=not IS_BINARY,  # Use softmax for multi-class, sigmoid for binary
        device=device,
    ).to(device)

    # Example input
    x = torch.randn(BATCH_SIZE, IN_CHANNELS, IMG_HEIGHT, IMG_WIDTH).to(device)

    print("Model testing...")

    # Test forward pass
    output = model(x)
    print(f"Output shape: {output.shape}")

    # Test confidence prediction
    pred, confidence, uncertainty = model.predict_with_confidence(x)
    print(f"Prediction shape: {pred.shape}")
    print(f"Confidence shape: {confidence.shape}")
    print(f"Uncertainty shape: {uncertainty.shape}")

    # Print model statistics
    print(f"\nModel weights: {model.model_weights.data}")
    print(f"Uncertainty weight: {model.uncertainty_weight.data}")
    print(f"Temperature: {model.temperature.data}")

    # Example training setup
    criterion = nn.BCEWithLogitsLoss() if IS_BINARY else nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)

    trainer = EnsembleTrainer(model, optimizer, criterion, device)
    print("\nEnsemble model ready for training!")
