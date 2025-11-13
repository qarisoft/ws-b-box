# import torch
# import torch.nn as nn
# import torch.nn.functional as F


# class SequentialBoostingEnsemble(nn.Module):
#     """
#     Sequential Boosting Ensemble for Medical Image Segmentation

#     This model implements a sequential boosting approach where three different segmentation
#     models (ENet, ResidualUNet, DeepLabV3) process the input in sequence, with each subsequent
#     model receiving guidance from the previous model's error maps. The final output is a
#     confidence-weighted fusion of all three models' predictions.

#     Key Features:
#     - Sequential processing with error propagation
#     - Confidence-based adaptive weighting
#     - Multi-stage refinement with error correction
#     - Feature concatenation and attention mechanisms

#     Architecture:
#     1. ENet (Stage 1): Fast, efficient initial segmentation
#     2. ResidualUNet (Stage 2): Refines output using Stage 1 error guidance
#     3. DeepLabV3 (Stage 3): Provides global context with Stage 2 error guidance
#     4. Fusion: Confidence-weighted combination of all stages

#     Args:
#         enet_model: Pre-trained ENet model instance
#         resunet_model: Pre-trained ResidualUNet model instance
#         deeplabv3_model: Pre-trained DeepLabV3 model instance
#         in_dim: Number of input channels (e.g., 1 for grayscale, 3 for RGB)
#         out_dim: Number of output classes/channels
#         softmax: Whether to use softmax activation (True for multi-class, False for binary)

#     Input:
#         x: Tensor of shape [batch_size, in_dim, height, width]

#     Output:
#         List containing single tensor of shape [batch_size, out_dim, height, width]
#     """

#     def __init__(
#         self,
#         enet_model,
#         resunet_model,
#         deeplabv3_model,
#         in_dim: int,
#         out_dim: int,
#         softmax: bool,
#     ):
#         super().__init__()
#         # Store individual models
#         self.enet = enet_model
#         self.resunet = resunet_model
#         self.deeplabv3 = deeplabv3_model
#         self.in_dim = in_dim
#         self.out_dim = out_dim
#         self.softmax = softmax

#         # Stage 1 Error Correction: Guides ResidualUNet based on ENet errors
#         # Processes concatenated original image and ENet error map
#         self.error_correction1 = nn.Sequential(
#             nn.Conv2d(
#                 out_dim + in_dim, 64, kernel_size=3, padding=1
#             ),  # Feature extraction
#             nn.BatchNorm2d(64),  # Normalization
#             nn.ReLU(inplace=True),  # Non-linearity
#             nn.Conv2d(64, in_dim, kernel_size=1),  # Project back to input dimensions
#         )

#         # Stage 2 Error Correction: Guides DeepLabV3 based on ResidualUNet errors
#         # Similar structure to Stage 1 correction
#         self.error_correction2 = nn.Sequential(
#             nn.Conv2d(out_dim + in_dim, 64, kernel_size=3, padding=1),
#             nn.BatchNorm2d(64),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(64, in_dim, kernel_size=1),
#         )

#         # Learnable confidence weights for each model
#         # Initialized equally, optimized during training
#         self.confidence_weights = nn.Parameter(torch.ones(3))

#         # Final Refinement: Combines features from all three models
#         # Processes concatenated predictions to produce refined output
#         self.final_refinement = nn.Sequential(
#             nn.Conv2d(out_dim * 3, 128, kernel_size=3, padding=1),  # Feature fusion
#             nn.BatchNorm2d(128),  # Batch normalization
#             nn.ReLU(inplace=True),  # Activation
#             nn.Conv2d(128, out_dim, kernel_size=1),  # Final projection
#         )

#         # Feature Attention: Learns spatial importance weights for different models
#         # Uses global average pooling and channel-wise attention
#         self.feature_attention = nn.Sequential(
#             nn.AdaptiveAvgPool2d(1),  # Global context
#             nn.Conv2d(out_dim * 3, out_dim, kernel_size=1),  # Channel weighting
#             nn.Sigmoid(),  # Normalize to [0,1]
#         )

#         print(
#             f"Initialized SequentialBoostingEnsemble with {in_dim} input channels and {out_dim} output channels"
#         )

#     def compute_confidence(self, predictions):
#         """
#         Compute confidence scores for model predictions

#         For multi-class (softmax): Uses entropy-based confidence
#         - Lower entropy = higher confidence (more certain predictions)
#         For binary (sigmoid): Uses distance from decision boundary
#         - Closer to 0 or 1 = higher confidence

#         Args:
#             predictions: Tensor of shape [B, C, H, W] with model predictions

#         Returns:
#             confidence: Tensor of shape [B, 1, H, W] with confidence scores
#         """
#         if self.softmax:
#             # Multi-class confidence: 1 / (1 + entropy)
#             # Entropy measures uncertainty in probability distribution
#             entropy = -torch.sum(predictions * torch.log(predictions + 1e-8), dim=1)
#             confidence = 1.0 / (1.0 + entropy.unsqueeze(1))
#         else:
#             # Binary confidence: max(p, 1-p)
#             # Measures distance from decision boundary (0.5)
#             confidence = torch.max(predictions, 1 - predictions, dim=1)[0].unsqueeze(1)
#         return confidence

#     def compute_error_map(self, predictions):
#         """
#         Compute error maps indicating prediction uncertainty

#         For multi-class: 1 - maximum class probability
#         For binary: 2 * |prediction - 0.5| (scaled to [0,1])

#         Args:
#             predictions: Tensor of shape [B, C, H, W] with model predictions

#         Returns:
#             error_map: Tensor of shape [B, 1, H, W] with error scores
#         """
#         if self.softmax:
#             # Error = 1 - maximum probability across classes
#             # High values indicate uncertain regions
#             max_prob, _ = torch.max(predictions, dim=1, keepdim=True)
#             error_map = 1.0 - max_prob
#         else:
#             # Error = distance from 0.5 (decision boundary)
#             # Scaled by 2 to range [0,1]
#             error_map = 2.0 * torch.abs(predictions - 0.5)
#         return error_map

#     def forward(self, x):
#         """
#         Forward pass through sequential boosting ensemble

#         Pipeline:
#         1. ENet → Error Map 1 → Corrected Input 2
#         2. ResidualUNet → Error Map 2 → Corrected Input 3
#         3. DeepLabV3 → Final Predictions
#         4. Confidence-weighted fusion + refinement

#         Args:
#             x: Input tensor [B, in_dim, H, W]

#         Returns:
#             List containing final segmentation [B, out_dim, H, W]
#         """
#         batch_size = x.shape[0]

#         # ========== STAGE 1: ENet (Fast, Efficient Initial Segmentation) ==========
#         # ENet provides quick, efficient segmentation with good edge detection
#         enet_output = self.enet(x)[0]  # [B, C, H, W]
#         enet_confidence = self.compute_confidence(enet_output)
#         enet_error = self.compute_error_map(enet_output)

#         # Prepare input for Stage 2: Original image + ENet error guidance
#         # Error map highlights regions where ENet is uncertain
#         stage2_input = torch.cat([x, enet_error], dim=1)
#         stage2_input = self.error_correction1(stage2_input)

#         # ========== STAGE 2: ResidualUNet (Balanced Refinement) ==========
#         # ResidualUNet refines segmentation with residual connections
#         # Focuses on regions where ENet was uncertain
#         resunet_output = self.resunet(stage2_input)[0]
#         resunet_confidence = self.compute_confidence(resunet_output)
#         resunet_error = self.compute_error_map(resunet_output)

#         # Prepare input for Stage 3: Original image + ResidualUNet error guidance
#         stage3_input = torch.cat([x, resunet_error], dim=1)
#         stage3_input = self.error_correction2(stage3_input)

#         # ========== STAGE 3: DeepLabV3 (Global Context Final Pass) ==========
#         # DeepLabV3 provides strong global context with ASPP module
#         # Handles complex structures and large receptive fields
#         deeplab_output = self.deeplabv3(stage3_input)[0]
#         deeplab_confidence = self.compute_confidence(deeplab_output)

#         # ========== CONFIDENCE-BASED FUSION ==========
#         # Compute average confidence for each model across spatial dimensions
#         confidences = torch.stack(
#             [
#                 enet_confidence.mean(),
#                 resunet_confidence.mean(),
#                 deeplab_confidence.mean(),
#             ]
#         )

#         # Combine learned weights with runtime confidence
#         normalized_weights = F.softmax(confidences, dim=0)
#         final_weights = F.softmax(self.confidence_weights * normalized_weights, dim=0)

#         # Weighted fusion of all three models
#         weighted_fusion = (
#             final_weights[0] * enet_output
#             + final_weights[1] * resunet_output
#             + final_weights[2] * deeplab_output
#         )

#         # ========== REFINEMENT FUSION ==========
#         # Alternative fusion: Concatenate all features and refine
#         concatenated_features = torch.cat(
#             [enet_output, resunet_output, deeplab_output], dim=1
#         )
#         refined_output = self.final_refinement(concatenated_features)

#         # ========== FINAL COMBINATION ==========
#         # Combine weighted average with refined features
#         # 70% refined output + 30% weighted fusion
#         final_output = 0.7 * refined_output + 0.3 * weighted_fusion

#         # Apply final activation function
#         if self.softmax:
#             final_output = F.softmax(final_output, dim=1)
#         else:
#             final_output = torch.sigmoid(final_output)

#         return [final_output]


# class ProgressiveBoostingEnsemble(nn.Module):
#     """
#     Progressive Boosting Ensemble with Residual Refinement

#     This model implements a progressive refinement approach where each model builds upon
#     the previous model's output. Unlike sequential boosting, this approach uses direct
#     refinement rather than error propagation, with residual connections to preserve
#     original predictions while adding improvements.

#     Key Features:
#     - Progressive refinement through stages
#     - Residual connections preserve original features
#     - Direct prediction refinement rather than error guidance
#     - Cumulative improvement across stages

#     Architecture:
#     1. ENet → Initial segmentation
#     2. ResidualUNet → Refines ENet output + original features
#     3. DeepLabV3 → Final refinement with global context
#     4. Weighted combination of all stages

#     Args:
#         enet_model: Pre-trained ENet model instance
#         resunet_model: Pre-trained ResidualUNet model instance
#         deeplabv3_model: Pre-trained DeepLabV3 model instance
#         in_dim: Number of input channels
#         out_dim: Number of output classes/channels
#         softmax: Whether to use softmax activation
#     """

#     def __init__(
#         self,
#         enet_model,
#         resunet_model,
#         deeplabv3_model,
#         in_dim: int,
#         out_dim: int,
#         softmax: bool,
#     ):
#         super().__init__()
#         self.enet = enet_model
#         self.resunet = resunet_model
#         self.deeplabv3 = deeplabv3_model
#         self.in_dim = in_dim
#         self.out_dim = out_dim
#         self.softmax = softmax

#         # Stage 1 Refinement: Refines ENet output using original image context
#         # Input: ENet predictions + original image
#         self.refinement1 = nn.Sequential(
#             nn.Conv2d(out_dim + in_dim, 64, kernel_size=3, padding=1),
#             nn.BatchNorm2d(64),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(64, out_dim, kernel_size=1),  # Output same as predictions
#         )

#         # Stage 2 Refinement: Combines Stage 1 output + ResidualUNet + original image
#         # More complex fusion with three input sources
#         self.refinement2 = nn.Sequential(
#             nn.Conv2d(out_dim * 2 + in_dim, 64, kernel_size=3, padding=1),
#             nn.BatchNorm2d(64),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(64, out_dim, kernel_size=1),
#         )

#         # Residual connections: Allow refinement to add improvements without losing original
#         self.residual_conv1 = nn.Conv2d(out_dim, out_dim, kernel_size=1)
#         self.residual_conv2 = nn.Conv2d(out_dim, out_dim, kernel_size=1)

#     def forward(self, x):
#         """
#         Forward pass through progressive refinement ensemble

#         Pipeline:
#         1. ENet → Initial segmentation → Refinement 1 → Stage 1 Final
#         2. ResidualUNet → Combined with Stage 1 → Refinement 2 → Stage 2 Final
#         3. DeepLabV3 → Final model → Weighted combination

#         Args:
#             x: Input tensor [B, in_dim, H, W]

#         Returns:
#             List containing final segmentation [B, out_dim, H, W]
#         """
#         # Stage 1: ENet - Fast initial segmentation with basic refinement
#         enet_out = self.enet(x)[0]

#         # Refine ENet output using original image context
#         # Combines segmentation with original features for context-aware refinement
#         stage1_refined = self.refinement1(torch.cat([enet_out, x], dim=1))
#         # Residual connection: original ENet + refinements
#         stage1_final = enet_out + self.residual_conv1(stage1_refined)

#         # Apply activation after each stage to maintain proper value ranges
#         if self.softmax:
#             stage1_final = F.softmax(stage1_final, dim=1)
#         else:
#             stage1_final = torch.sigmoid(stage1_final)

#         # Stage 2: ResidualUNet - Intermediate refinement with more capacity
#         resunet_out = self.resunet(x)[0]

#         # Combine Stage 1 output with ResidualUNet and original image
#         # Three-way fusion: previous stage + current model + original context
#         stage2_input = torch.cat([stage1_final, resunet_out, x], dim=1)
#         stage2_refined = self.refinement2(stage2_input)
#         # Residual connection: original ResidualUNet + multi-source refinements
#         stage2_final = resunet_out + self.residual_conv2(stage2_refined)

#         if self.softmax:
#             stage2_final = F.softmax(stage2_final, dim=1)
#         else:
#             stage2_final = torch.sigmoid(stage2_final)

#         # Stage 3: DeepLabV3 - Final pass with global context
#         deeplab_out = self.deeplabv3(x)[0]

#         # Adaptive fusion: Weighted combination of all three stages
#         # Weights: Stage1 (20%), Stage2 (30%), Stage3 (50%)
#         # DeepLabV3 gets highest weight due to its strong global context
#         final_output = stage1_final * 0.2 + stage2_final * 0.3 + deeplab_out * 0.5

#         # Final activation
#         if self.softmax:
#             final_output = F.softmax(final_output, dim=1)
#         else:
#             final_output = torch.sigmoid(final_output)

#         return [final_output]


# class AdaptiveBoostingEnsemble(nn.Module):
#     """
#     Adaptive Boosting Ensemble with Spatial Attention

#     This model implements an adaptive fusion approach that learns spatial attention maps
#     to determine which model to trust in different image regions. Unlike sequential or
#     progressive approaches, this model processes all three models in parallel and uses
#     attention mechanisms to combine their predictions spatially.

#     Key Features:
#     - Parallel processing of all three models
#     - Spatial attention for region-specific model selection
#     - Dynamic weighting across image regions
#     - Confidence estimation for quality assessment

#     Architecture:
#     1. Parallel: ENet, ResidualUNet, DeepLabV3 process input simultaneously
#     2. Attention: Generate spatial attention maps for each model
#     3. Fusion: Apply attention weights to model predictions
#     4. Combination: Sum attended predictions for final output

#     Args:
#         enet_model: Pre-trained ENet model instance
#         resunet_model: Pre-trained ResidualUNet model instance
#         deeplabv3_model: Pre-trained DeepLabV3 model instance
#         in_dim: Number of input channels
#         out_dim: Number of output classes/channels
#         softmax: Whether to use softmax activation
#     """

#     def __init__(
#         self,
#         enet_model,
#         resunet_model,
#         deeplabv3_model,
#         in_dim: int,
#         out_dim: int,
#         softmax: bool,
#     ):
#         super().__init__()
#         self.enet = enet_model
#         self.resunet = resunet_model
#         self.deeplabv3 = deeplabv3_model
#         self.in_dim = in_dim
#         self.out_dim = out_dim
#         self.softmax = softmax

#         # Spatial Attention: Learns which model to trust in each spatial location
#         # Input: Concatenated predictions from all three models
#         # Output: 3 attention maps (one for each model) that sum to 1 at each location
#         self.spatial_attention = nn.Sequential(
#             nn.Conv2d(out_dim * 3, 64, kernel_size=3, padding=1),  # Feature extraction
#             nn.BatchNorm2d(64),  # Normalization
#             nn.ReLU(inplace=True),  # Non-linearity
#             nn.Conv2d(64, 3, kernel_size=1),  # 3 attention channels
#             nn.Softmax(dim=1),  # Normalize to sum=1 per spatial location
#         )

#         # Confidence Estimation: Optional module to estimate prediction confidence
#         # Can be used for uncertainty quantification or quality control
#         self.confidence_net = nn.Sequential(
#             nn.Conv2d(out_dim, 32, kernel_size=3, padding=1),
#             nn.BatchNorm2d(32),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(32, 1, kernel_size=1),
#             nn.Sigmoid(),  # Confidence scores in [0,1]
#         )

#     def forward(self, x):
#         """
#         Forward pass through adaptive attention ensemble

#         Pipeline:
#         1. Parallel processing by all three models
#         2. Concatenate all predictions
#         3. Generate spatial attention maps
#         4. Apply attention weights to each model's predictions
#         5. Sum attended predictions for final output

#         Args:
#             x: Input tensor [B, in_dim, H, W]

#         Returns:
#             List containing final segmentation [B, out_dim, H, W]
#         """
#         # Parallel processing: All models process the input simultaneously
#         # This is more computationally efficient than sequential processing
#         enet_out = self.enet(x)[0]  # ENet: Good for edges and efficiency
#         resunet_out = self.resunet(x)[
#             0
#         ]  # ResidualUNet: Good balance of detail and context
#         deeplab_out = self.deeplabv3(x)[0]  # DeepLabV3: Strong global context

#         # Concatenate all predictions along channel dimension
#         # Creates rich feature set with diverse segmentation characteristics
#         all_outputs = torch.cat([enet_out, resunet_out, deeplab_out], dim=1)

#         # Generate spatial attention weights
#         # Each spatial location gets 3 weights that sum to 1
#         # These weights determine how much to trust each model at each location
#         attention_weights = self.spatial_attention(all_outputs)

#         # Apply attention weights to each model's predictions
#         # Element-wise multiplication: Each model's output is weighted spatially
#         attended_enet = enet_out * attention_weights[:, 0:1, :, :]  # ENet attention
#         attended_resunet = (
#             resunet_out * attention_weights[:, 1:2, :, :]
#         )  # ResidualUNet attention
#         attended_deeplab = (
#             deeplab_out * attention_weights[:, 2:3, :, :]
#         )  # DeepLabV3 attention

#         # Combine attended features by summation
#         # This creates the final segmentation where each location is dominated by
#         # the model that the attention mechanism trusts most there
#         final_output = attended_enet + attended_resunet + attended_deeplab

#         # Apply final activation function
#         if self.softmax:
#             final_output = F.softmax(final_output, dim=1)
#         else:
#             final_output = torch.sigmoid(final_output)

#         return [final_output]


# def create_sequential_boosting(
#     enet_model,
#     resunet_model,
#     deeplabv3_model,
#     in_dim: int,
#     out_dim: int,
#     softmax: bool,
#     model_type: str = "sequential",
# ):
#     """
#     Factory function to create sequential boosting ensemble models

#     Provides a unified interface for creating different types of boosting ensembles
#     with consistent parameter handling and initialization.

#     Available Model Types:
#     - 'sequential': SequentialBoostingEnsemble - Error propagation through stages
#     - 'progressive': ProgressiveBoostingEnsemble - Direct refinement with residuals
#     - 'adaptive': AdaptiveBoostingEnsemble - Spatial attention-based fusion

#     Args:
#         enet_model: Pre-trained ENet model instance
#         resunet_model: Pre-trained ResidualUNet model instance
#         deeplabv3_model: Pre-trained DeepLabV3 model instance
#         in_dim: Number of input channels (1 for grayscale, 3 for RGB)
#         out_dim: Number of output classes/channels
#         softmax: Whether to use softmax activation (True for multi-class, False for binary)
#         model_type: Type of boosting ensemble ('sequential', 'progressive', 'adaptive')

#     Returns:
#         Initialized boosting ensemble model instance

#     Raises:
#         ValueError: If unknown model_type is provided

#     Example:
#         >>> model = create_sequential_boosting(enet, resunet, deeplab,
#         ...                                   in_dim=1, out_dim=2, softmax=True,
#         ...                                   model_type='sequential')
#     """
#     if model_type == "sequential":
#         return SequentialBoostingEnsemble(
#             enet_model, resunet_model, deeplabv3_model, in_dim, out_dim, softmax
#         )
#     elif model_type == "progressive":
#         return ProgressiveBoostingEnsemble(
#             enet_model, resunet_model, deeplabv3_model, in_dim, out_dim, softmax
#         )
#     elif model_type == "adaptive":
#         return AdaptiveBoostingEnsemble(
#             enet_model, resunet_model, deeplabv3_model, in_dim, out_dim, softmax
#         )
#     else:
#         raise ValueError(
#             f"Unknown model type: {model_type}. "
#             f"Available types: 'sequential', 'progressive', 'adaptive'"
#         )


# # Example usage and testing
# if __name__ == "__main__":
#     """
#     Example demonstration of sequential boosting ensemble creation and usage

#     This example shows how to initialize the individual models and create
#     a boosting ensemble for medical image segmentation.
#     """

#     # Assuming model classes are defined elsewhere
#     # from models import ENet, ResidualUNet, DeepLabV3

#     # Initialize individual models with consistent parameters
#     enet = ENet(in_dim=1, out_dim=2, softmax=True)
#     resunet = ResidualUNet(in_dim=1, out_dim=2, softmax=True)
#     deeplabv3 = DeepLabV3(in_dim=1, out_dim=2, softmax=True)

#     # Create sequential boosting model
#     boosting_model = create_sequential_boosting(
#         enet_model=enet,
#         resunet_model=resunet,
#         deeplabv3_model=deeplabv3,
#         in_dim=1,
#         out_dim=2,
#         softmax=True,
#         model_type="sequential",
#     )

#     # Test the model with sample input
#     test_input = torch.randn(2, 1, 256, 256)  # Batch of 2, 1 channel, 256x256 images
#     output = boosting_model(test_input)

#     print(f"Input shape: {test_input.shape}")
#     print(f"Output shape: {output[0].shape}")
#     print(
#         f"Number of parameters: {sum(p.numel() for p in boosting_model.parameters()):,}"
#     )
