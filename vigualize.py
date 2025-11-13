import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch


def debug_gt_visualization(data_root, split="train", max_images=4):
    """
    Debug function to see what's happening with the images
    """
    print("🐛 DEBUG MODE - Checking image loading and display...")

    img_dir = os.path.join(data_root, split, "img")
    gt_dir = os.path.join(data_root, split, "gt")

    print(f"📁 Image dir: {img_dir}")
    print(f"📁 GT dir: {gt_dir}")

    # Find image files
    image_files = glob.glob(os.path.join(img_dir, "*.*"))
    print(f"📸 Found {len(image_files)} image files")

    if not image_files:
        print("❌ No images found!")
        return

    # Take first few images
    test_files = image_files[:max_images]

    for i, img_path in enumerate(test_files):
        print(f"\n🔍 Processing image {i+1}: {os.path.basename(img_path)}")

        # Find corresponding GT
        img_stem = os.path.splitext(os.path.basename(img_path))[0]
        gt_path = None
        for ext in [".png", ".jpg", ".jpeg", ".tif", ".tiff"]:
            possible_gt = os.path.join(gt_dir, f"{img_stem}{ext}")
            if os.path.exists(possible_gt):
                gt_path = possible_gt
                break

        if not gt_path:
            print(f"❌ No GT found for {img_stem}")
            continue

        print(f"✅ Found GT: {os.path.basename(gt_path)}")

        try:
            # Load images
            img = Image.open(img_path).convert("RGB")
            gt = Image.open(gt_path).convert("L")  # Force grayscale

            # Convert to numpy
            img_np = np.array(img)
            gt_np = np.array(gt)

            print(
                f"   Image shape: {img_np.shape}, dtype: {img_np.dtype}, range: [{img_np.min()}, {img_np.max()}]"
            )
            print(
                f"   GT shape: {gt_np.shape}, dtype: {gt_np.dtype}, range: [{gt_np.min()}, {gt_np.max()}]"
            )

            # Check if GT is all zeros
            if gt_np.max() == 0:
                print("   ⚠️  WARNING: GT is all zeros (completely black)!")
            elif gt_np.max() == 1:
                print("   ℹ️  GT appears to be binary (0s and 1s)")
            elif gt_np.max() == 255:
                print("   ℹ️  GT appears to be 8-bit (0-255)")

            # Create simple visualization
            fig, axes = plt.subplots(1, 4, figsize=(16, 4))

            # Original image
            axes[0].imshow(img_np)
            axes[0].set_title("Original Image")
            axes[0].axis("off")

            # GT as grayscale
            axes[1].imshow(gt_np, cmap="gray")
            axes[1].set_title("GT (Grayscale)")
            axes[1].axis("off")

            # GT with viridis colormap
            im = axes[2].imshow(gt_np, cmap="viridis")
            axes[2].set_title("GT (Viridis)")
            axes[2].axis("off")
            plt.colorbar(im, ax=axes[2])

            # GT histogram
            axes[3].hist(gt_np.flatten(), bins=50, alpha=0.7)
            axes[3].set_title("GT Value Distribution")
            axes[3].set_xlabel("Pixel Value")
            axes[3].set_ylabel("Frequency")

            plt.tight_layout()
            plt.show()

        except Exception as e:
            print(f"❌ Error: {e}")


def simple_gt_overlay(data_root, split="train", image_index=0):
    """
    Simple function to create overlay with enhanced GT visibility
    """
    print("🎨 Creating enhanced overlay...")

    img_dir = os.path.join(data_root, split, "img")
    gt_dir = os.path.join(data_root, split, "gt")

    # Get specific image
    image_files = glob.glob(os.path.join(img_dir, "*.*"))
    if not image_files:
        print("❌ No images found!")
        return

    img_path = image_files[image_index]
    img_stem = os.path.splitext(os.path.basename(img_path))[0]

    # Find GT
    gt_path = None
    for ext in [".png", ".jpg", ".jpeg", ".tif", ".tiff"]:
        possible_gt = os.path.join(gt_dir, f"{img_stem}{ext}")
        if os.path.exists(possible_gt):
            gt_path = possible_gt
            break

    if not gt_path:
        print(f"❌ No GT found for {img_stem}")
        return

    print(f"🖼️  Image: {os.path.basename(img_path)}")
    print(f"🎯 GT: {os.path.basename(gt_path)}")

    try:
        # Load images
        img = Image.open(img_path).convert("RGB")
        gt = Image.open(gt_path).convert("L")

        img_np = np.array(img)
        gt_np = np.array(gt)

        print(f"📊 Image: {img_np.shape}, range: [{img_np.min()}, {img_np.max()}]")
        print(f"📊 GT: {gt_np.shape}, range: [{gt_np.min()}, {gt_np.max()}]")

        # Enhanced GT processing
        gt_enhanced = enhance_gt_visibility(gt_np)

        # Create multiple overlay attempts
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))

        # Row 1: Basic displays
        axes[0, 0].imshow(img_np)
        axes[0, 0].set_title("Original Image")
        axes[0, 0].axis("off")

        axes[0, 1].imshow(gt_np, cmap="gray")
        axes[0, 1].set_title("Original GT (Grayscale)")
        axes[0, 1].axis("off")

        axes[0, 2].imshow(gt_enhanced, cmap="viridis")
        axes[0, 2].set_title("Enhanced GT (Viridis)")
        axes[0, 2].axis("off")

        # Row 2: Different overlay attempts
        overlay1 = create_bright_overlay(img_np, gt_np, color="red", alpha=0.8)
        axes[1, 0].imshow(overlay1)
        axes[1, 0].set_title("Red Overlay (α=0.8)")
        axes[1, 0].axis("off")

        overlay2 = create_bright_overlay(img_np, gt_np, color="green", alpha=1.0)
        axes[1, 1].imshow(overlay2)
        axes[1, 1].set_title("Green Overlay (α=1.0)")
        axes[1, 1].axis("off")

        overlay3 = create_bright_overlay(img_np, gt_enhanced, color="yellow", alpha=0.9)
        axes[1, 2].imshow(overlay3)
        axes[1, 2].set_title("Yellow Overlay (Enhanced GT)")
        axes[1, 2].axis("off")

        plt.tight_layout()
        plt.show()

    except Exception as e:
        print(f"❌ Error: {e}")


def enhance_gt_visibility(gt_np):
    """
    Enhance GT visibility by scaling and thresholding
    """
    # Normalize to [0, 1]
    gt_normalized = (gt_np - gt_np.min()) / (gt_np.max() - gt_np.min() + 1e-8)

    # If GT has very small values, scale them up
    if gt_normalized.max() < 0.1:
        print("   🔍 GT values are very small, scaling up...")
        gt_enhanced = gt_normalized * 10  # Scale up
        gt_enhanced = np.clip(gt_enhanced, 0, 1)
    else:
        gt_enhanced = gt_normalized

    # Apply threshold to make faint regions more visible
    threshold = 0.1
    gt_enhanced[gt_enhanced > threshold] = 1.0
    gt_enhanced[gt_enhanced <= threshold] = 0.0

    return gt_enhanced


def create_bright_overlay(img_np, gt_np, color="red", alpha=0.8):
    """
    Create bright, highly visible overlay
    """
    # Ensure img is RGB
    if len(img_np.shape) == 2:
        img_rgb = np.stack([img_np] * 3, axis=2)
    else:
        img_rgb = img_np

    # Normalize both to [0, 1]
    img_normalized = img_rgb.astype(np.float32) / 255.0
    gt_normalized = (gt_np - gt_np.min()) / (gt_np.max() - gt_np.min() + 1e-8)

    # Create bright colored mask
    colored_mask = np.zeros_like(img_normalized)

    if color == "red":
        colored_mask[:, :, 0] = 1.0  # Bright red
    elif color == "green":
        colored_mask[:, :, 1] = 1.0  # Bright green
    elif color == "blue":
        colored_mask[:, :, 2] = 1.0  # Bright blue
    elif color == "yellow":
        colored_mask[:, :, 0] = 1.0  # Red
        colored_mask[:, :, 1] = 1.0  # Green

    # Apply GT as mask
    colored_mask = colored_mask * gt_normalized[:, :, np.newaxis]

    # Create overlay
    overlay = img_normalized * (1 - alpha) + colored_mask * alpha
    overlay = np.clip(overlay, 0, 1)

    return overlay


def check_directory_contents(data_root):
    """
    Check what's actually in the directories
    """
    print("📁 Checking directory contents...")

    for split in ["train", "val"]:
        img_dir = os.path.join(data_root, split, "img")
        gt_dir = os.path.join(data_root, split, "gt")

        print(f"\n{split.upper()}:")
        if os.path.exists(img_dir):
            img_files = os.listdir(img_dir)
            print(f"   img/ : {len(img_files)} files")
            if img_files:
                print(f"   First 5: {img_files[:5]}")
        else:
            print(f"   img/ : ❌ Directory not found!")

        if os.path.exists(gt_dir):
            gt_files = os.listdir(gt_dir)
            print(f"   gt/  : {len(gt_files)} files")
            if gt_files:
                print(f"   First 5: {gt_files[:5]}")
        else:
            print(f"   gt/  : ❌ Directory not found!")


# Run these functions to debug:
if __name__ == "__main__":
    data_root = "./data/prostate/PROSTATE"  # Change this to your actual data path

    print("🚀 Starting GT Visualization Debug...")
    print("=" * 50)

    # First, check what's in the directories
    check_directory_contents(data_root)

    print("\n" + "=" * 50)
    print("🐛 Running debug visualization...")

    # Run debug visualization
    debug_gt_visualization(data_root, split="train", max_images=2)

    print("\n" + "=" * 50)
    print("🎨 Trying enhanced overlay...")

    # Try enhanced overlay
    simple_gt_overlay(data_root, split="train", image_index=0)
# import os
# import glob
# import torch
# import numpy as np
# import matplotlib.pyplot as plt
# from pathlib import Path
# from PIL import Image, ImageFilter, ImageEnhance
# import torchvision.transforms as transforms


# def visualize_gt_from_directory(
#     data_root,
#     split="train",
#     max_images=12,
#     save_dir=None,
#     show=True,
#     figsize=(20, 15),
#     overlay_alpha=0.6,
#     overlay_color="green",
#     cmap="viridis",
# ):
#     """
#     Visualize ground truth images from directory structure with overlays

#     Directory structure expected:
#     data_root/
#     ├── train/
#     │   ├── img/
#     │   └── gt/
#     └── val/
#         ├── img/
#         └── gt/

#     Args:
#         data_root: Root directory containing train/val folders
#         split: Which split to visualize ('train', 'val', or 'both')
#         max_images: Maximum number of images to display
#         save_dir: Directory to save visualizations
#         show: Whether to display the plot
#         figsize: Figure size for matplotlib
#         overlay_alpha: Transparency for GT overlay
#         overlay_color: Color for GT overlay ('green', 'red', 'blue', 'yellow', 'cyan', 'magenta')
#         cmap: Colormap for standalone GT display
#     """

#     # Create save directory if specified
#     if save_dir:
#         os.makedirs(save_dir, exist_ok=True)

#     # Determine which splits to process
#     if split == "both":
#         splits = ["train", "val"]
#     else:
#         splits = [split]

#     all_image_gt_pairs = []

#     for current_split in splits:
#         img_dir = os.path.join(data_root, current_split, "img")
#         gt_dir = os.path.join(data_root, current_split, "gt")

#         if not os.path.exists(img_dir):
#             print(f"❌ Image directory not found: {img_dir}")
#             continue
#         if not os.path.exists(gt_dir):
#             print(f"❌ GT directory not found: {gt_dir}")
#             continue

#         # Find all image files
#         image_extensions = [".jpg", ".png", ".jpeg", ".tif", ".tiff", ".bmp"]
#         image_files = []
#         for ext in image_extensions:
#             image_files.extend(glob.glob(os.path.join(img_dir, f"*{ext}")))
#             image_files.extend(glob.glob(os.path.join(img_dir, f"*{ext.upper()}")))

#         if not image_files:
#             print(f"❌ No images found in {img_dir}")
#             continue

#         print(f"📁 Found {len(image_files)} images in {current_split}/img")

#         # Find corresponding GT files
#         for img_path in image_files:
#             img_name = os.path.basename(img_path)
#             img_stem = os.path.splitext(img_name)[0]

#             # Look for GT file with same name (different extensions)
#             gt_path = None
#             for ext in image_extensions:
#                 possible_gt_path = os.path.join(gt_dir, f"{img_stem}{ext}")
#                 if os.path.exists(possible_gt_path):
#                     gt_path = possible_gt_path
#                     break
#                 # Also check with different case
#                 possible_gt_path_upper = os.path.join(
#                     gt_dir, f"{img_stem}{ext.upper()}"
#                 )
#                 if os.path.exists(possible_gt_path_upper):
#                     gt_path = possible_gt_path_upper
#                     break

#             if gt_path:
#                 all_image_gt_pairs.append((img_path, gt_path, current_split))
#             else:
#                 print(f"⚠️  No GT found for {img_name}")

#     if not all_image_gt_pairs:
#         print("❌ No image-GT pairs found!")
#         return

#     print(f"🎯 Found {len(all_image_gt_pairs)} image-GT pairs")

#     # Limit number of images
#     display_pairs = all_image_gt_pairs[:max_images]

#     # Calculate grid layout
#     ncols = 4  # Original, GT, Overlay, Difference
#     nrows = len(display_pairs)

#     fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
#     if nrows == 1:
#         axes = axes.reshape(1, -1)

#     for row_idx, (img_path, gt_path, split_name) in enumerate(display_pairs):
#         try:
#             # Load image using PIL
#             img = load_image_pil(img_path)
#             gt = load_image_pil(gt_path, grayscale=True)

#             # Convert to numpy for processing
#             img_np = np.array(img)
#             gt_np = np.array(gt)

#             # Handle different image dimensions
#             if img_np.ndim == 3:
#                 img_np = img_np.transpose(2, 0, 1)  # [H, W, C] -> [C, H, W]
#             if gt_np.ndim == 3:
#                 gt_np = gt_np.mean(axis=2)  # Convert to grayscale if RGB

#             # Normalize images for display
#             img_display = normalize_image_np(img_np)
#             gt_display = normalize_image_np(gt_np)

#             # Create overlay
#             overlay = create_colored_overlay_pil(
#                 img_display, gt_display, color=overlay_color, alpha=overlay_alpha
#             )

#             # Create difference (for visualization)
#             if img_display.ndim == 3:
#                 img_gray = img_display.mean(axis=0)
#             else:
#                 img_gray = img_display
#             difference = np.abs(img_gray - gt_display)

#             # Plot 1: Original image
#             ax1 = axes[row_idx, 0]
#             if img_display.ndim == 3:
#                 ax1.imshow(img_display.transpose(1, 2, 0))  # [C, H, W] -> [H, W, C]
#             else:
#                 ax1.imshow(img_display, cmap="gray")
#             ax1.set_title(
#                 f"{split_name}\nOriginal: {os.path.basename(img_path)}", fontsize=10
#             )
#             ax1.axis("off")

#             # Plot 2: Ground truth
#             ax2 = axes[row_idx, 1]
#             im2 = ax2.imshow(gt_display, cmap=cmap)
#             ax2.set_title(f"Ground Truth\n{os.path.basename(gt_path)}", fontsize=10)
#             ax2.axis("off")
#             # Add colorbar for GT
#             plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

#             # Plot 3: Overlay
#             ax3 = axes[row_idx, 2]
#             ax3.imshow(overlay)
#             ax3.set_title(
#                 f"Overlay ({overlay_color})\nAlpha: {overlay_alpha}", fontsize=10
#             )
#             ax3.axis("off")

#             # Plot 4: Difference
#             ax4 = axes[row_idx, 3]
#             im4 = ax4.imshow(difference, cmap="hot")
#             ax4.set_title("Difference\n|Image - GT|", fontsize=10)
#             ax4.axis("off")
#             plt.colorbar(im4, ax=ax4, fraction=0.046, pad=0.04)

#             # Print image info
#             print(
#                 f"📊 {split_name}/{os.path.basename(img_path)} - "
#                 f"Img: {img_np.shape}, GT: {gt_np.shape}, "
#                 f"GT range: [{gt_np.min():.3f}, {gt_np.max():.3f}]"
#             )

#         except Exception as e:
#             print(f"❌ Error processing {img_path}: {e}")
#             # Clear the axes for this row
#             for col_idx in range(4):
#                 axes[row_idx, col_idx].clear()
#                 axes[row_idx, col_idx].text(
#                     0.5,
#                     0.5,
#                     f"Error\n{str(e)[:30]}...",
#                     ha="center",
#                     va="center",
#                     transform=axes[row_idx, col_idx].transAxes,
#                 )
#                 axes[row_idx, col_idx].axis("off")

#     plt.tight_layout()

#     # Save if requested
#     if save_dir:
#         save_path = os.path.join(save_dir, f"gt_visualization_{split}.png")
#         plt.savefig(save_path, dpi=150, bbox_inches="tight")
#         print(f"💾 Visualization saved to: {save_path}")

#     if show:
#         plt.show()
#     else:
#         plt.close()


# def load_image_pil(image_path, grayscale=False):
#     """
#     Load image from path using PIL
#     """
#     try:
#         if grayscale:
#             image = Image.open(image_path).convert("L")  # Convert to grayscale
#         else:
#             image = Image.open(image_path).convert("RGB")

#         return image

#     except Exception as e:
#         print(f"❌ Error loading image {image_path}: {e}")
#         # Return a black image as fallback
#         if grayscale:
#             return Image.new("L", (256, 256), 0)
#         else:
#             return Image.new("RGB", (256, 256), (0, 0, 0))


# def normalize_image_np(image):
#     """
#     Normalize image to [0, 1] range for display
#     """
#     if image.dtype == np.uint8:
#         image = image.astype(np.float32) / 255.0
#     else:
#         image = (image - image.min()) / (image.max() - image.min() + 1e-8)

#     return image


# def create_colored_overlay_pil(original, mask, color="green", alpha=0.6):
#     """
#     Create colored overlay of mask on original image using PIL/numpy

#     Args:
#         original: Original image array [C, H, W] or [H, W]
#         mask: Mask image array [H, W]
#         color: Overlay color
#         alpha: Transparency
#     """
#     # Ensure original is 3 channels [H, W, 3]
#     if original.ndim == 3:
#         if original.shape[0] == 3:  # [C, H, W] -> [H, W, C]
#             original_rgb = original.transpose(1, 2, 0)
#         else:  # [H, W, C] already
#             original_rgb = original
#     else:  # Grayscale [H, W]
#         original_rgb = np.stack([original] * 3, axis=2)

#     # Ensure mask is 2D
#     if mask.ndim == 3:
#         mask = mask.squeeze()

#     # Normalize mask
#     mask_normalized = (mask - mask.min()) / (mask.max() - mask.min() + 1e-8)

#     # Create colored mask based on chosen color
#     colored_mask = np.zeros_like(original_rgb)

#     if color == "green":
#         colored_mask[:, :, 1] = mask_normalized  # Green channel
#     elif color == "red":
#         colored_mask[:, :, 0] = mask_normalized  # Red channel
#     elif color == "blue":
#         colored_mask[:, :, 2] = mask_normalized  # Blue channel
#     elif color == "yellow":
#         colored_mask[:, :, 0] = mask_normalized  # Red
#         colored_mask[:, :, 1] = mask_normalized  # Green
#     elif color == "cyan":
#         colored_mask[:, :, 1] = mask_normalized  # Green
#         colored_mask[:, :, 2] = mask_normalized  # Blue
#     elif color == "magenta":
#         colored_mask[:, :, 0] = mask_normalized  # Red
#         colored_mask[:, :, 2] = mask_normalized  # Blue
#     else:
#         colored_mask[:, :, 1] = mask_normalized  # Default to green

#     # Create overlay
#     overlay = original_rgb * (1 - alpha) + colored_mask * alpha
#     overlay = np.clip(overlay, 0, 1)

#     return overlay


# def visualize_single_pair_pil(
#     img_path,
#     gt_path,
#     save_path=None,
#     show=True,
#     figsize=(15, 5),
#     overlay_alpha=0.6,
#     overlay_color="green",
# ):
#     """
#     Visualize a single image-GT pair in detail using PIL
#     """
#     fig, axes = plt.subplots(1, 4, figsize=figsize)

#     try:
#         # Load images using PIL
#         img = load_image_pil(img_path)
#         gt = load_image_pil(gt_path, grayscale=True)

#         # Convert to numpy
#         img_np = np.array(img)
#         gt_np = np.array(gt)

#         # Handle dimensions
#         if img_np.ndim == 3:
#             img_np = img_np.transpose(2, 0, 1)  # [H, W, C] -> [C, H, W]
#         if gt_np.ndim == 3:
#             gt_np = gt_np.mean(axis=2)  # Convert to grayscale if RGB

#         # Normalize
#         img_display = normalize_image_np(img_np)
#         gt_display = normalize_image_np(gt_np)

#         # Create overlay
#         overlay = create_colored_overlay_pil(
#             img_display, gt_display, color=overlay_color, alpha=overlay_alpha
#         )

#         # Create difference
#         if img_display.ndim == 3:
#             img_gray = img_display.mean(axis=0)
#         else:
#             img_gray = img_display
#         difference = np.abs(img_gray - gt_display)

#         # Plot 1: Original image
#         axes[0].imshow(
#             img_display.transpose(1, 2, 0) if img_display.ndim == 3 else img_display,
#             cmap=None if img_display.ndim == 3 else "gray",
#         )
#         axes[0].set_title(f"Original Image\n{os.path.basename(img_path)}")
#         axes[0].axis("off")

#         # Plot 2: Ground truth
#         im1 = axes[1].imshow(gt_display, cmap="viridis")
#         axes[1].set_title(f"Ground Truth\n{os.path.basename(gt_path)}")
#         axes[1].axis("off")
#         plt.colorbar(im1, ax=axes[1])

#         # Plot 3: Overlay
#         axes[2].imshow(overlay)
#         axes[2].set_title(f"Overlay ({overlay_color}, α={overlay_alpha})")
#         axes[2].axis("off")

#         # Plot 4: Difference
#         im3 = axes[3].imshow(difference, cmap="hot")
#         axes[3].set_title("Difference Map")
#         axes[3].axis("off")
#         plt.colorbar(im3, ax=axes[3])

#         # Add statistics
#         stats_text = f"Image: {img_np.shape}, GT: {gt_np.shape}, GT range: [{gt_np.min():.3f}, {gt_np.max():.3f}]"
#         fig.text(
#             0.5,
#             0.01,
#             stats_text,
#             ha="center",
#             fontsize=12,
#             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"),
#         )

#     except Exception as e:
#         for ax in axes:
#             ax.clear()
#             ax.text(
#                 0.5,
#                 0.5,
#                 f"Error: {str(e)}",
#                 ha="center",
#                 va="center",
#                 transform=ax.transAxes,
#             )
#             ax.axis("off")

#     plt.tight_layout()

#     if save_path:
#         plt.savefig(save_path, dpi=150, bbox_inches="tight")
#         print(f"💾 Single pair saved to: {save_path}")

#     if show:
#         plt.show()
#     else:
#         plt.close()


# def apply_image_filters_pil(image, filter_type="sharpen", factor=1.5):
#     """
#     Apply various filters to PIL image for enhancement
#     """
#     if filter_type == "sharpen":
#         return image.filter(ImageFilter.SHARPEN)
#     elif filter_type == "contrast":
#         enhancer = ImageEnhance.Contrast(image)
#         return enhancer.enhance(factor)
#     elif filter_type == "brightness":
#         enhancer = ImageEnhance.Brightness(image)
#         return enhancer.enhance(factor)
#     elif filter_type == "edge_enhance":
#         return image.filter(ImageFilter.EDGE_ENHANCE)
#     else:
#         return image


# def analyze_dataset_structure(data_root):
#     """
#     Analyze and print dataset structure
#     """
#     print("🔍 Analyzing dataset structure...")
#     print(f"📁 Root: {data_root}")

#     for split in ["train", "val"]:
#         img_dir = os.path.join(data_root, split, "img")
#         gt_dir = os.path.join(data_root, split, "gt")

#         if os.path.exists(img_dir) and os.path.exists(gt_dir):
#             img_files = glob.glob(os.path.join(img_dir, "*.*"))
#             gt_files = glob.glob(os.path.join(gt_dir, "*.*"))

#             print(f"\n📊 {split.upper()} Split:")
#             print(f"   Images: {len(img_files)} files in {img_dir}")
#             print(f"   GT: {len(gt_files)} files in {gt_dir}")

#             # Check file correspondence
#             img_names = {os.path.splitext(os.path.basename(f))[0] for f in img_files}
#             gt_names = {os.path.splitext(os.path.basename(f))[0] for f in gt_files}

#             matched = img_names & gt_names
#             img_only = img_names - gt_names
#             gt_only = gt_names - img_names

#             print(f"   ✅ Matched pairs: {len(matched)}")
#             if img_only:
#                 print(f"   ⚠️  Images without GT: {len(img_only)}")
#             if gt_only:
#                 print(f"   ⚠️  GT without images: {len(gt_only)}")
#         else:
#             print(f"\n❌ {split.upper()} Split: Missing directories")


# # Example usage
# def demonstrate_gt_visualization_pil(data_root, output_dir="./gt_visualizations"):
#     """
#     Demonstrate all visualization functions using PIL
#     """
#     print("🎨 Demonstrating GT Visualization from Directory (PIL version)...")

#     os.makedirs(output_dir, exist_ok=True)

#     # Analyze dataset structure
#     analyze_dataset_structure(data_root)

#     # Visualize training data
#     visualize_gt_from_directory(
#         data_root=data_root,
#         split="train",
#         max_images=8,
#         save_dir=output_dir,
#         show=False,
#         overlay_color="green",
#         overlay_alpha=0.6,
#     )

#     # Visualize validation data
#     visualize_gt_from_directory(
#         data_root=data_root,
#         split="val",
#         max_images=8,
#         save_dir=output_dir,
#         show=False,
#         overlay_color="red",
#         overlay_alpha=0.5,
#     )

#     # Find and visualize a single pair in detail
#     train_img_dir = os.path.join(data_root, "train", "img")
#     train_gt_dir = os.path.join(data_root, "train", "gt")

#     if os.path.exists(train_img_dir) and os.path.exists(train_gt_dir):
#         img_files = glob.glob(os.path.join(train_img_dir, "*.*"))
#         if img_files:
#             img_path = img_files[0]
#             img_stem = os.path.splitext(os.path.basename(img_path))[0]

#             # Find corresponding GT
#             for ext in [".png", ".jpg", ".jpeg", ".tif", ".tiff"]:
#                 gt_path = os.path.join(train_gt_dir, f"{img_stem}{ext}")
#                 if os.path.exists(gt_path):
#                     visualize_single_pair_pil(
#                         img_path,
#                         gt_path,
#                         save_path=os.path.join(output_dir, "single_pair_detail.png"),
#                         show=False,
#                     )
#                     break

#     print(f"✅ All visualizations saved to: {output_dir}")


# # Quick test
# if __name__ == "__main__":
#     # Test with a sample directory structure
#     test_data_root = "./test_data"

#     # Create test directory structure
#     os.makedirs(os.path.join(test_data_root, "train", "img"), exist_ok=True)
#     os.makedirs(os.path.join(test_data_root, "train", "gt"), exist_ok=True)
#     os.makedirs(os.path.join(test_data_root, "val", "img"), exist_ok=True)
#     os.makedirs(os.path.join(test_data_root, "val", "gt"), exist_ok=True)

#     print("🧪 Created test directory structure")
#     print("💡 Now put your actual images in:")
#     print(f"   {test_data_root}/train/img/")
#     print(f"   {test_data_root}/train/gt/")
#     print(f"   {test_data_root}/val/img/")
#     print(f"   {test_data_root}/val/gt/")

#     # Test with actual data if available
#     if os.path.exists("./data"):
#         visualize_gt_from_directory(
#             data_root="./data.ful/prostate/PROSTATE",
#             max_images=12,
#             save_dir="./visualizations",
#         )
