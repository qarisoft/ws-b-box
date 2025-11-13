import math
import sys
import time
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
from torchvision_wj.utils import utils
import matplotlib.pyplot as plt
import numpy as np
import os


class TensorBoardLogger:
    """TensorBoard logger for training and validation metrics and images"""

    def __init__(self, log_dir):
        self.writer = SummaryWriter(log_dir=log_dir)
        self.global_step = 0

    def log_scalars(self, scalar_dict, step, prefix=""):
        """Log scalar values to TensorBoard"""
        for key, value in scalar_dict.items():
            if value is not None:
                full_key = f"{prefix}/{key}" if prefix else key
                self.writer.add_scalar(full_key, value, step)

    def log_images(self, images_dict, step, prefix=""):
        """Log images to TensorBoard"""
        for key, image_tensor in images_dict.items():
            if image_tensor is not None:
                full_key = f"{prefix}/{key}" if prefix else key

                # Ensure image is in correct format [C, H, W] and normalized
                if image_tensor.dim() == 2:  # [H, W]
                    image_tensor = image_tensor.unsqueeze(0)  # [1, H, W]
                elif (
                    image_tensor.dim() == 3
                    and image_tensor.shape[0] != 1
                    and image_tensor.shape[0] != 3
                ):
                    image_tensor = image_tensor.permute(
                        2, 0, 1
                    )  # [H, W, C] -> [C, H, W]

                # Normalize to [0, 1] if needed
                if image_tensor.max() > 1.0:
                    image_tensor = (image_tensor - image_tensor.min()) / (
                        image_tensor.max() - image_tensor.min()
                    )

                self.writer.add_image(full_key, image_tensor, step)

    def log_image_grid(self, images_list, step, prefix="", nrow=4):
        """Log a grid of images to TensorBoard"""
        if images_list:
            # Stack images and create grid
            images_tensor = torch.stack(images_list)
            grid = make_grid(images_tensor, nrow=nrow, normalize=True, scale_each=True)
            full_key = f"{prefix}/image_grid" if prefix else "image_grid"
            self.writer.add_image(full_key, grid, step)

    def close(self):
        """Close the TensorBoard writer"""
        self.writer.close()


def train_one_epoch(
    model,
    optimizer,
    data_loader,
    device,
    epoch,
    clipnorm=0.001,
    print_freq=20,
    tb_logger=None,
    save_images_freq=100,
):
    """
    Train one epoch with TensorBoard logging and image saving

    Args:
        tb_logger: TensorBoardLogger instance
        save_images_freq: How often to save images (in batches)
    """
    time.sleep(2)  # Prevent possible deadlock during epoch transition
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = "Epoch: [{}]".format(epoch)

    lr_scheduler = None
    if epoch == 0:
        warmup_factor = 1.0 / 1000
        warmup_iters = min(1000, len(data_loader) - 1)
        lr_scheduler = utils.warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor)

    for batch_idx, (images, targets) in enumerate(
        metric_logger.log_every(data_loader, print_freq, header)
    ):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict, seg_preds = model(images, targets)  # Get segmentation predictions

        losses = sum(loss for loss in loss_dict.values())

        # Log training images to TensorBoard
        if tb_logger and batch_idx % save_images_freq == 0:
            log_training_images(images, targets, seg_preds, epoch, batch_idx, tb_logger)

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())
        loss_value = losses_reduced.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        if clipnorm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clipnorm)
        optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()

        metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        # Log batch metrics to TensorBoard
        if tb_logger:
            tb_logger.global_step += 1
            batch_metrics = {
                "train_loss": losses_reduced.item(),
                "learning_rate": optimizer.param_groups[0]["lr"],
            }
            # Add individual losses
            for loss_name, loss_val in loss_dict_reduced.items():
                batch_metrics[f"train_{loss_name}"] = loss_val.item()

            tb_logger.log_scalars(batch_metrics, tb_logger.global_step, prefix="batch")

    # Log epoch metrics to TensorBoard
    if tb_logger:
        epoch_metrics = {}
        for name, meter in metric_logger.meters.items():
            if name == "lr":
                epoch_metrics["learning_rate"] = meter.global_avg
            else:
                epoch_metrics[f"train_{name}"] = meter.global_avg

        tb_logger.log_scalars(epoch_metrics, epoch, prefix="epoch")

    return metric_logger


def log_training_images(images, targets, seg_preds, epoch, batch_idx, tb_logger):
    """Log training images to TensorBoard"""
    try:
        # Take first image from batch for visualization
        image = images[0].cpu().detach()
        target_mask = targets[0]["masks"].cpu().detach()
        pred_mask = seg_preds[0][0].cpu().detach()  # Assuming seg_preds is a list

        # Prepare images for TensorBoard
        images_dict = {}

        # Original image
        if image.shape[0] == 3:  # RGB
            image_display = image
        else:  # Grayscale
            image_display = image.repeat(
                3, 1, 1
            )  # Convert to 3-channel for consistent display

        images_dict["input_image"] = image_display

        # Ground truth mask (convert to 3-channel)
        if len(target_mask.shape) == 3:  # [C, H, W]
            gt_mask_display = target_mask[0].unsqueeze(0).repeat(3, 1, 1)
        else:
            gt_mask_display = target_mask.repeat(3, 1, 1)
        images_dict["ground_truth"] = gt_mask_display

        # Prediction mask (convert to 3-channel)
        if len(pred_mask.shape) == 3:  # [C, H, W]
            pred_mask_display = pred_mask[0].unsqueeze(0).repeat(3, 1, 1)
        else:
            pred_mask_display = pred_mask.repeat(3, 1, 1)
        images_dict["prediction"] = pred_mask_display

        # Overlay
        overlay = create_overlay(
            image_display, pred_mask_display[0]
        )  # Use first channel
        images_dict["overlay"] = overlay

        # Log to TensorBoard
        tb_logger.log_images(images_dict, tb_logger.global_step, prefix="train")

        # Also create and log a comparison grid
        comparison_images = [image_display, gt_mask_display, pred_mask_display, overlay]
        tb_logger.log_image_grid(
            comparison_images, tb_logger.global_step, prefix="train_comparison", nrow=4
        )

    except Exception as e:
        print(f"Warning: Failed to log training images: {e}")


@torch.no_grad()
def validate_loss(
    model, data_loader, device, tb_logger=None, epoch=0, save_images_freq=10
):
    """
    Validate model with TensorBoard logging and image saving

    Args:
        tb_logger: TensorBoardLogger instance
        save_images_freq: How often to save images (in batches)
    """
    n_threads = torch.get_num_threads()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Validation: "

    # FIXME remove this and make paste_masks_in_image run on the GPU
    torch.set_num_threads(1)
    cpu_device = torch.device("cpu")
    model.eval()

    import pandas as pd

    loss_summary = []
    all_val_images = []  # Store images for grid

    for batch_idx, (images, targets) in enumerate(
        metric_logger.log_every(
            data_loader, print_freq=10e5, header=header, training=False
        )
    ):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict, seg_preds = model(images, targets)

        losses = sum(loss for loss in loss_dict.values())

        # Log validation images to TensorBoard
        if tb_logger and batch_idx % save_images_freq == 0:
            val_images = log_validation_images(
                images, targets, seg_preds, epoch, batch_idx, tb_logger
            )
            all_val_images.extend(val_images)

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())
        loss_dict_reduced = dict(
            ("val_" + k, f(v) if hasattr(v, "keys") else v)
            for k, v in loss_dict_reduced.items()
        )

        loss_value = losses_reduced.item()
        metric_logger.update(val_loss=losses_reduced, **loss_dict_reduced)

        loss_reduced = dict(
            (k, f(v) if hasattr(v, "keys") else v.item())
            for k, v in loss_dict_reduced.items()
        )
        loss_reduced.update(dict(gt=targets[0]["boxes"].shape[0]))
        loss_summary.append(loss_reduced)

    # Log validation metrics to TensorBoard
    if tb_logger:
        val_metrics = {}
        for name, meter in metric_logger.meters.items():
            val_metrics[name] = meter.global_avg

        tb_logger.log_scalars(val_metrics, epoch, prefix="validation")

        # Log validation image grid
        if all_val_images:
            tb_logger.log_image_grid(
                all_val_images[:16], epoch, prefix="validation_grid", nrow=4
            )

    loss_summary = pd.DataFrame(loss_summary)
    loss_summary.to_csv("val_image_summary.csv", index=False)

    torch.set_num_threads(n_threads)

    return metric_logger


def log_validation_images(images, targets, seg_preds, epoch, batch_idx, tb_logger):
    """Log validation images to TensorBoard and return images for grid"""
    val_images = []

    try:
        # Take first image from batch for visualization
        image = images[0].cpu().detach()
        target_mask = targets[0]["masks"].cpu().detach()
        pred_mask = seg_preds[0][0].cpu().detach()

        # Prepare images for TensorBoard
        images_dict = {}

        # Original image
        if image.shape[0] == 3:  # RGB
            image_display = image
        else:  # Grayscale
            image_display = image.repeat(3, 1, 1)

        images_dict["input_image"] = image_display
        val_images.append(image_display)

        # Ground truth mask
        if len(target_mask.shape) == 3:
            gt_mask_display = target_mask[0].unsqueeze(0).repeat(3, 1, 1)
        else:
            gt_mask_display = target_mask.repeat(3, 1, 1)
        images_dict["ground_truth"] = gt_mask_display
        val_images.append(gt_mask_display)

        # Prediction mask
        if len(pred_mask.shape) == 3:
            pred_mask_display = pred_mask[0].unsqueeze(0).repeat(3, 1, 1)
        else:
            pred_mask_display = pred_mask.repeat(3, 1, 1)
        images_dict["prediction"] = pred_mask_display
        val_images.append(pred_mask_display)

        # Overlay
        overlay = create_overlay(image_display, pred_mask_display[0])
        images_dict["overlay"] = overlay
        val_images.append(overlay)

        # Log to TensorBoard
        tb_logger.log_images(images_dict, epoch * 1000 + batch_idx, prefix="validation")

    except Exception as e:
        print(f"Warning: Failed to log validation images: {e}")

    return val_images


def create_overlay(original_image, mask, alpha=0.5):
    """
    Create an overlay of original image and mask

    Args:
        original_image: Tensor [3, H, W]
        mask: Tensor [H, W] or [1, H, W]
        alpha: Transparency for mask overlay
    """
    # Ensure mask is single channel and normalized
    if mask.dim() == 3:
        mask = mask[0]  # Take first channel

    # Normalize mask to [0, 1]
    mask = (mask - mask.min()) / (mask.max() - mask.min() + 1e-8)

    # Create colored mask (red for segmentation)
    colored_mask = torch.zeros_like(original_image)
    colored_mask[0] = mask  # Red channel

    # Create overlay
    overlay = original_image * (1 - alpha) + colored_mask * alpha

    return overlay


# Simple standalone test
def test_tensorboard_logging():
    """Test TensorBoard logging functionality"""

    # Create a simple model and dummy data for testing
    class DummyModel(torch.nn.Module):
        def forward(self, images, targets=None):
            # Return dummy losses and predictions
            if targets is None:
                return {}, [torch.randn(1, 1, 256, 256) for _ in images]

            losses = {"loss1": torch.tensor(0.5), "loss2": torch.tensor(0.3)}
            preds = [torch.randn(1, 1, 256, 256) for _ in images]
            return losses, preds

    # Test TensorBoard logger
    tb_logger = TensorBoardLogger(log_dir="./test_tensorboard")

    # Test image logging
    dummy_image = torch.randn(3, 256, 256)
    dummy_mask = torch.randn(1, 256, 256)

    tb_logger.log_images(
        {"test_input": dummy_image, "test_mask": dummy_mask.repeat(3, 1, 1)}, step=0
    )

    tb_logger.close()
    print("TensorBoard test completed. Run: tensorboard --logdir=./test_tensorboard")


if __name__ == "__main__":
    test_tensorboard_logging()
