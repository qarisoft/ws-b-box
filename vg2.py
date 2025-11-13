import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# Load image and ground truth data
img_path = "./data/prostate/PROSTATE/train/img/Case01_0_17.png"
mask_path = "./data/prostate/PROSTATE/train/gt/Case01_0_17.png"
image = np.array(Image.open(img_path))
mask = np.array(Image.open(mask_path))

# # Create subplots
# fig, axes = plt.subplots(1, 2, figsize=(12, 6))
# ax1, ax2 = axes.flatten()

# # Display original image
# ax1.imshow(image)
# ax1.set_title("Original Image")
# ax1.axis("off")

# # Display ground truth
# ax2.imshow(image)  # Use 'gray' if mask is binary/grayscale
# ax2.imshow(mask, cmap="gray")  # Use 'gray' if mask is binary/grayscale
# ax2.set_title("Ground Truth")
# ax2.axis("off")

# # Adjust layout and show
# plt.tight_layout()
# plt.show()


import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# Load image (ensure RGB)
# img_path = 'your_image.jpg'
image = np.array(Image.open(img_path))

# Load or generate ground truth mask (2D numpy array with values 0 or 1, etc.)
# The mask should have the same height and width as the image.
# Example mask (replace with your actual mask loading logic):
# mask = np.zeros(image.shape[:2], dtype=np.uint8)
# mask[100:300, 50:200] = 1
# mask_path = 'your_mask.png' # Or load a mask image
mask = np.array(Image.open(mask_path))

# Create figure and axes
fig, ax = plt.subplots(1)
ax.imshow(image)

# Overlay the mask with transparency
# Use a colormap and alpha value for visualization
ax.imshow(mask, cmap="jet", alpha=0.5)  # 'jet' colormap and 0.5 transparency

# Hide axis ticks
ax.axis("off")

# Show the plot
plt.title("Image with Ground Truth Segmentation Mask Overlay")
plt.show()
