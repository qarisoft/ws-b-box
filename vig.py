# look at my example i want you to make gt img like my example to be show over the img and the gt to be in diifrent color

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

img_path = "./data/prostate/PROSTATE/train/img/Case01_0_17.png"
mask_path = "./data/prostate/PROSTATE/train/gt/Case01_0_17.png"
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


import tkinter as tk
from tkinter import ttk, filedialog
from PIL import Image, ImageTk, ImageDraw, ImageOps
import os
import glob


class ImageGTViewer:
    def __init__(self, root):
        self.root = root
        self.root.title("Image and Ground Truth Viewer")
        self.root.geometry("1400x900")

        # Data structure
        self.data_path = "./data.ful/prostate/PROSTATE"
        self.current_set = "train"  # or "val"
        self.image_files = []
        self.gt_files = []
        self.current_index = 0

        # Overlay settings
        self.overlay_color = (255, 0, 0)  # Red for GT overlay
        self.overlay_alpha = 0.5  # Transparency level

        # Create UI
        self.create_widgets()

    def create_widgets(self):
        # Top frame for controls
        control_frame = ttk.Frame(self.root, padding="10")
        control_frame.pack(fill=tk.X)

        # Data directory selection
        ttk.Label(control_frame, text="Data Directory:").grid(
            row=0, column=0, sticky=tk.W, padx=(0, 10)
        )
        self.dir_var = tk.StringVar()
        ttk.Entry(control_frame, textvariable=self.dir_var, width=50).grid(
            row=0, column=1, padx=(0, 10)
        )
        ttk.Button(control_frame, text="Browse", command=self.browse_directory).grid(
            row=0, column=2
        )

        # Dataset selection
        ttk.Label(control_frame, text="Dataset:").grid(
            row=1, column=0, sticky=tk.W, padx=(0, 10), pady=(10, 0)
        )
        self.dataset_var = tk.StringVar(value="train")
        dataset_combo = ttk.Combobox(
            control_frame,
            textvariable=self.dataset_var,
            values=["train", "val"],
            state="readonly",
        )
        dataset_combo.grid(row=1, column=1, sticky=tk.W, padx=(0, 10), pady=(10, 0))
        dataset_combo.bind("<<ComboboxSelected>>", self.on_dataset_change)

        # File navigation
        ttk.Label(control_frame, text="File:").grid(
            row=2, column=0, sticky=tk.W, padx=(0, 10), pady=(10, 0)
        )
        self.file_var = tk.StringVar()
        self.file_combo = ttk.Combobox(
            control_frame, textvariable=self.file_var, state="readonly", width=50
        )
        self.file_combo.grid(row=2, column=1, sticky=tk.W, padx=(0, 10), pady=(10, 0))
        self.file_combo.bind("<<ComboboxSelected>>", self.on_file_select)

        # Overlay controls
        ttk.Label(control_frame, text="Overlay Color:").grid(
            row=0, column=3, sticky=tk.W, padx=(20, 10)
        )
        self.color_var = tk.StringVar(value="Red")
        color_combo = ttk.Combobox(
            control_frame,
            textvariable=self.color_var,
            values=["Red", "Green", "Blue", "Yellow", "Cyan", "Magenta"],
            state="readonly",
            width=10,
        )
        color_combo.grid(row=0, column=4, sticky=tk.W, padx=(0, 10))
        color_combo.bind("<<ComboboxSelected>>", self.on_color_change)

        ttk.Label(control_frame, text="Opacity:").grid(
            row=1, column=3, sticky=tk.W, padx=(20, 10), pady=(10, 0)
        )
        self.alpha_var = tk.DoubleVar(value=0.5)
        alpha_scale = ttk.Scale(
            control_frame,
            from_=0.1,
            to=1.0,
            variable=self.alpha_var,
            orient=tk.HORIZONTAL,
            length=100,
        )
        alpha_scale.grid(row=1, column=4, sticky=tk.W, padx=(0, 10), pady=(10, 0))
        alpha_scale.bind("<ButtonRelease-1>", self.on_alpha_change)

        # Main display area with three panels
        display_frame = ttk.Frame(self.root)
        display_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Create three frames for the three displays
        self.frame_original = ttk.LabelFrame(display_frame, text="Original Image")
        self.frame_original.pack(
            side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5
        )

        self.frame_gt = ttk.LabelFrame(display_frame, text="Ground Truth")
        self.frame_gt.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.frame_overlay = ttk.LabelFrame(display_frame, text="Overlay (Image + GT)")
        self.frame_overlay.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Canvases for each frame
        self.canvas_original = tk.Canvas(
            self.frame_original, bg="white", relief=tk.SUNKEN, borderwidth=1
        )
        self.canvas_original.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.canvas_gt = tk.Canvas(
            self.frame_gt, bg="white", relief=tk.SUNKEN, borderwidth=1
        )
        self.canvas_gt.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.canvas_overlay = tk.Canvas(
            self.frame_overlay, bg="white", relief=tk.SUNKEN, borderwidth=1
        )
        self.canvas_overlay.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Navigation buttons
        nav_frame = ttk.Frame(self.root)
        nav_frame.pack(fill=tk.X, padx=10, pady=10)

        ttk.Button(nav_frame, text="First", command=self.first_image).pack(
            side=tk.LEFT, padx=5
        )
        ttk.Button(nav_frame, text="Previous", command=self.previous_image).pack(
            side=tk.LEFT, padx=5
        )
        ttk.Button(nav_frame, text="Next", command=self.next_image).pack(
            side=tk.LEFT, padx=5
        )
        ttk.Button(nav_frame, text="Last", command=self.last_image).pack(
            side=tk.LEFT, padx=5
        )

        # Status bar
        self.status_var = tk.StringVar(value="Please select a data directory")
        status_bar = ttk.Label(
            self.root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W
        )
        status_bar.pack(fill=tk.X, side=tk.BOTTOM)

    def browse_directory(self):
        directory = filedialog.askdirectory(title="Select Data Directory")
        if directory:
            self.dir_var.set(directory)
            self.data_path = directory
            self.load_dataset()

    def load_dataset(self):
        if not self.data_path:
            return

        self.current_set = self.dataset_var.get()
        img_dir = os.path.join(self.data_path, self.current_set, "img")
        gt_dir = os.path.join(self.data_path, self.current_set, "gt")

        # Check if directories exist
        if not os.path.exists(img_dir) or not os.path.exists(gt_dir):
            self.status_var.set(
                f"Error: 'img' or 'gt' folder not found in {self.current_set}"
            )
            return

        # Get image files
        self.image_files = []
        for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tiff"]:
            self.image_files.extend(glob.glob(os.path.join(img_dir, ext)))
            self.image_files.extend(glob.glob(os.path.join(img_dir, ext.upper())))

        if not self.image_files:
            self.status_var.set(f"No images found in {img_dir}")
            return

        # Sort files for consistent ordering
        self.image_files.sort()

        # Update file selector
        file_names = [os.path.basename(f) for f in self.image_files]
        self.file_combo["values"] = file_names

        # Load first image
        self.current_index = 0
        self.file_var.set(file_names[0])
        self.display_image()

    def on_dataset_change(self, event):
        self.load_dataset()

    def on_file_select(self, event):
        selected_file = self.file_var.get()
        if selected_file and self.image_files:
            file_names = [os.path.basename(f) for f in self.image_files]
            if selected_file in file_names:
                self.current_index = file_names.index(selected_file)
                self.display_image()

    def on_color_change(self, event):
        color_map = {
            "Red": (255, 0, 0),
            "Green": (0, 255, 0),
            "Blue": (0, 0, 255),
            "Yellow": (255, 255, 0),
            "Cyan": (0, 255, 255),
            "Magenta": (255, 0, 255),
        }
        self.overlay_color = color_map.get(self.color_var.get(), (255, 0, 0))
        self.display_image()

    def on_alpha_change(self, event):
        self.overlay_alpha = self.alpha_var.get()
        self.display_image()

    def display_image(self):
        if not self.image_files or self.current_index >= len(self.image_files):
            return

        # Load original image
        img_path = self.image_files[self.current_index]
        img_name = os.path.basename(img_path)

        # Find corresponding GT image
        gt_dir = os.path.join(self.data_path, self.current_set, "gt")
        gt_name, gt_ext = os.path.splitext(img_name)

        # Try to find GT with same name (different extensions possible)
        gt_candidates = []
        for ext in [".png", ".jpg", ".jpeg", ".bmp", ".tiff"]:
            candidate = os.path.join(gt_dir, gt_name + ext)
            if os.path.exists(candidate):
                gt_candidates.append(candidate)
            # Also try with same extension as original
            candidate = os.path.join(gt_dir, img_name)
            if os.path.exists(candidate):
                gt_candidates.append(candidate)

        if not gt_candidates:
            self.status_var.set(f"No ground truth found for {img_name}")
            # Display only the original image in all panels
            try:
                img = Image.open(img_path).convert("RGB")
                self.display_on_canvas(self.canvas_original, img, "Original Image")
                self.display_on_canvas(self.canvas_gt, img, "GT Not Found")
                self.display_on_canvas(
                    self.canvas_overlay, img, "Overlay Not Available"
                )
            except Exception as e:
                self.status_var.set(f"Error loading image: {str(e)}")
            return

        # Use the first found GT image
        gt_path = gt_candidates[0]

        try:
            # Load images
            img = Image.open(img_path).convert("RGB")
            gt = Image.open(gt_path)

            # Convert GT to binary mask if needed and enhance for display
            if gt.mode != "1":
                # Threshold to create binary mask
                gt_binary = gt.convert("L")
                # Use a threshold to create binary image
                threshold = 128
                gt_binary = gt_binary.point(lambda p: p > threshold and 255)
            else:
                gt_binary = gt.convert("L")

            # Create a visually enhanced version of GT for display
            gt_display = self.enhance_gt_for_display(gt_binary)

            # Create overlay
            overlay = self.create_overlay(img, gt_binary)

            # Display all three images
            self.display_on_canvas(self.canvas_original, img, "Original Image")
            self.display_on_canvas(self.canvas_gt, gt_display, "Ground Truth")
            self.display_on_canvas(self.canvas_overlay, overlay, "Overlay")

            # Update status
            self.status_var.set(
                f"Displaying {img_name} ({self.current_index+1}/{len(self.image_files)})"
            )

        except Exception as e:
            self.status_var.set(f"Error processing images: {str(e)}")

    def enhance_gt_for_display(self, gt_image):
        """Enhance the ground truth image for better visualization"""
        # Create a colored version of the GT for display
        gt_rgb = Image.new("RGB", gt_image.size, (0, 0, 0))

        # Convert GT to binary mask (0 or 255)
        gt_binary = gt_image.point(lambda p: 255 if p > 128 else 0)

        # Create a colored version (white on black background)
        colored_gt = Image.new("RGB", gt_image.size, (0, 0, 0))
        # Make the GT regions white
        colored_gt.paste((255, 255, 255), (0, 0), gt_binary)

        return colored_gt

    def create_overlay(self, img, gt):
        """Create an overlay of the image with GT on top"""
        # Create overlay
        overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
        mask = gt.convert("L")

        # Create colored overlay
        color_overlay = Image.new(
            "RGBA", img.size, self.overlay_color + (int(255 * self.overlay_alpha),)
        )

        # Apply the GT as mask to the color overlay
        overlay = Image.composite(color_overlay, overlay, mask)

        # Convert base image to RGBA for compositing
        img_rgba = img.convert("RGBA")

        # Combine image and overlay
        result = Image.alpha_composite(img_rgba, overlay).convert("RGB")

        return result

    def display_on_canvas(self, canvas, image, title):
        # Clear canvas
        canvas.delete("all")

        # Get canvas dimensions
        canvas_width = canvas.winfo_width()
        canvas_height = canvas.winfo_height()

        if canvas_width <= 1 or canvas_height <= 1:
            # Canvas not yet rendered, use default size
            canvas_width = 400
            canvas_height = 400

        # Calculate scaling to fit image in canvas while maintaining aspect ratio
        img_width, img_height = image.size
        scale = (
            min(canvas_width / img_width, canvas_height / img_height) * 0.9
        )  # 10% margin

        if scale < 1.0 or scale > 1.0:  # Always resize to fit canvas nicely
            new_width = int(img_width * scale)
            new_height = int(img_height * scale)
            image_resized = image.resize(
                (new_width, new_height), Image.Resampling.LANCZOS
            )
        else:
            image_resized = image
            new_width = img_width
            new_height = img_height

        # Convert to PhotoImage for display
        tk_image = ImageTk.PhotoImage(image_resized)

        # Store reference to prevent garbage collection
        if canvas == self.canvas_original:
            self.tk_image_original = tk_image
        elif canvas == self.canvas_gt:
            self.tk_image_gt = tk_image
        else:
            self.tk_image_overlay = tk_image

        # Calculate position to center the image
        x = (canvas_width - new_width) // 2
        y = (canvas_height - new_height) // 2

        # Display image
        canvas.create_image(x, y, anchor=tk.NW, image=tk_image)

        # Add title at the bottom
        canvas.create_text(
            canvas_width // 2,
            canvas_height - 15,
            text=title,
            fill="black",
            font=("Arial", 12, "bold"),
        )

    def first_image(self):
        if self.image_files:
            self.current_index = 0
            self.file_var.set(os.path.basename(self.image_files[0]))
            self.display_image()

    def previous_image(self):
        if self.image_files and self.current_index > 0:
            self.current_index -= 1
            self.file_var.set(os.path.basename(self.image_files[self.current_index]))
            self.display_image()

    def next_image(self):
        if self.image_files and self.current_index < len(self.image_files) - 1:
            self.current_index += 1
            self.file_var.set(os.path.basename(self.image_files[self.current_index]))
            self.display_image()

    def last_image(self):
        if self.image_files:
            self.current_index = len(self.image_files) - 1
            self.file_var.set(os.path.basename(self.image_files[self.current_index]))
            self.display_image()


if __name__ == "__main__":
    root = tk.Tk()
    app = ImageGTViewer(root)
    root.mainloop()
