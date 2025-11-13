import tkinter as tk
from tkinter import ttk, filedialog
from PIL import Image, ImageTk, ImageDraw, ImageOps
import os
import glob
import numpy as np


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
        self.overlay_alpha = 0.5  # Transparency level
        self.colormap = "jet"  # Default colormap

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

        # Colormap controls
        ttk.Label(control_frame, text="Colormap:").grid(
            row=0, column=3, sticky=tk.W, padx=(20, 10)
        )
        self.colormap_var = tk.StringVar(value="jet")
        colormap_combo = ttk.Combobox(
            control_frame,
            textvariable=self.colormap_var,
            values=[
                "jet",
                "viridis",
                "plasma",
                "hot",
                "cool",
                "spring",
                "summer",
                "autumn",
                "winter",
            ],
            state="readonly",
            width=10,
        )
        colormap_combo.grid(row=0, column=4, sticky=tk.W, padx=(0, 10))
        colormap_combo.bind("<<ComboboxSelected>>", self.on_colormap_change)

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

    def on_colormap_change(self, event):
        self.colormap = self.colormap_var.get()
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

            # Convert images to numpy arrays
            img_array = np.array(img)
            gt_array = np.array(gt)

            # Process GT array
            if len(gt_array.shape) == 3:
                gt_array = gt_array[:, :, 0]  # Take first channel if multi-channel

            # Normalize GT to 0-1 range
            if gt_array.max() > 0:
                gt_normalized = gt_array.astype(np.float32) / gt_array.max()
            else:
                gt_normalized = gt_array.astype(np.float32)

            # Create colormap overlay (similar to matplotlib)
            colored_gt = self.apply_colormap(gt_normalized, self.colormap)

            # Create overlay similar to matplotlib style
            overlay = self.create_matplotlib_style_overlay(
                img_array, gt_normalized, self.colormap
            )

            # Convert back to PIL images for display
            gt_display = Image.fromarray(colored_gt)
            overlay_display = Image.fromarray(overlay)

            # Display all three images
            self.display_on_canvas(self.canvas_original, img, "Original Image")
            self.display_on_canvas(self.canvas_gt, gt_display, "Ground Truth")
            self.display_on_canvas(self.canvas_overlay, overlay_display, "Overlay")

            # Update status
            self.status_var.set(
                f"Displaying {img_name} ({self.current_index+1}/{len(self.image_files)}) - Colormap: {self.colormap}"
            )

        except Exception as e:
            self.status_var.set(f"Error processing images: {str(e)}")

    def apply_colormap(self, data, cmap_name):
        """Apply colormap to data similar to matplotlib"""
        # Normalize data to 0-255
        if data.max() > 0:
            data_normalized = (data * 255).astype(np.uint8)
        else:
            data_normalized = (data * 255).astype(np.uint8)

        # Apply colormap
        if cmap_name == "jet":
            return self.jet_colormap(data_normalized)
        elif cmap_name == "viridis":
            return self.viridis_colormap(data_normalized)
        elif cmap_name == "plasma":
            return self.plasma_colormap(data_normalized)
        elif cmap_name == "hot":
            return self.hot_colormap(data_normalized)
        elif cmap_name == "cool":
            return self.cool_colormap(data_normalized)
        else:
            return self.jet_colormap(data_normalized)

    def jet_colormap(self, data):
        """Jet colormap similar to matplotlib"""
        # Simple implementation of jet colormap
        result = np.zeros((data.shape[0], data.shape[1], 3), dtype=np.uint8)

        # Red channel
        result[:, :, 0] = np.clip(255 * (1.5 - 4 * np.abs(data / 255 - 0.75)), 0, 255)
        # Green channel
        result[:, :, 1] = np.clip(255 * (1.5 - 4 * np.abs(data / 255 - 0.5)), 0, 255)
        # Blue channel
        result[:, :, 2] = np.clip(255 * (1.5 - 4 * np.abs(data / 255 - 0.25)), 0, 255)

        return result

    def viridis_colormap(self, data):
        """Viridis colormap approximation"""
        # Simplified viridis-like colors
        x = data / 255.0
        result = np.zeros((data.shape[0], data.shape[1], 3), dtype=np.uint8)

        # Viridis-like color progression
        result[:, :, 0] = (68 + (188 - 68) * x).astype(np.uint8)  # R
        result[:, :, 1] = (1 + (189 - 1) * x).astype(np.uint8)  # G
        result[:, :, 2] = (84 + (157 - 84) * x**0.8).astype(np.uint8)  # B

        return result

    def plasma_colormap(self, data):
        """Plasma colormap approximation"""
        x = data / 255.0
        result = np.zeros((data.shape[0], data.shape[1], 3), dtype=np.uint8)

        # Plasma-like color progression
        result[:, :, 0] = (13 + (240 - 13) * x).astype(np.uint8)  # R
        result[:, :, 1] = (8 + (249 - 8) * (1 - x) ** 1.5).astype(np.uint8)  # G
        result[:, :, 2] = (135 + (33 - 135) * x).astype(np.uint8)  # B

        return result

    def hot_colormap(self, data):
        """Hot colormap"""
        result = np.zeros((data.shape[0], data.shape[1], 3), dtype=np.uint8)

        # Hot colormap: black -> red -> yellow -> white
        x = data / 255.0
        r = np.clip(3 * x, 0, 1)
        g = np.clip(3 * x - 1, 0, 1)
        b = np.clip(3 * x - 2, 0, 1)

        result[:, :, 0] = (r * 255).astype(np.uint8)
        result[:, :, 1] = (g * 255).astype(np.uint8)
        result[:, :, 2] = (b * 255).astype(np.uint8)

        return result

    def cool_colormap(self, data):
        """Cool colormap"""
        result = np.zeros((data.shape[0], data.shape[1], 3), dtype=np.uint8)

        # Cool colormap: cyan -> magenta
        x = data / 255.0
        result[:, :, 0] = (x * 255).astype(np.uint8)  # R
        result[:, :, 1] = ((1 - x) * 255).astype(np.uint8)  # G
        result[:, :, 2] = 255  # B

        return result

    def create_matplotlib_style_overlay(self, img_array, gt_normalized, cmap_name):
        """Create overlay similar to matplotlib's imshow with transparency"""
        # Apply colormap to GT
        colored_gt = self.apply_colormap(
            (gt_normalized * 255).astype(np.uint8), cmap_name
        )

        # Create overlay with transparency
        overlay = img_array.copy().astype(np.float32)

        # Only apply overlay where GT has values
        mask = gt_normalized > 0.1

        for c in range(3):
            overlay_channel = overlay[:, :, c]
            gt_channel = colored_gt[:, :, c].astype(np.float32)

            # Blend: original * (1-alpha) + colored_gt * alpha
            overlay_channel[mask] = (
                overlay_channel[mask] * (1 - self.overlay_alpha)
                + gt_channel[mask] * self.overlay_alpha
            )
            overlay[:, :, c] = overlay_channel

        return overlay.astype(np.uint8)

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
