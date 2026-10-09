import tkinter as tk
from tkinter import ttk

from pathlib import Path
from PIL import Image, ImageTk


class PreviewResolver:

    IMAGE_EXTENSIONS = {
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".tif",
        ".tiff",
        ".webp"
    }

    # Prioritas hasil yang ingin ditampilkan.
    #
    # Kalau segmentation ada,
    # tampilkan segmentation.
    #
    # Kalau tidak ada,
    # fallback ke enhancement,
    # lalu preprocessing,
    # lalu detection.

    STAGES = [
        "03_segmentation",
        "02_enhancement",
        "01_preprocessing",
        "00_detection"
    ]

    def __init__(
        self,
        output_dir="output"
    ):

        self.output_dir = Path(
            output_dir
        )

    def find_original_and_processed(
        self,
        experiment_name,
        image_name
    ):

        if " - " in image_name:
            parts = image_name.split(" - ")
            folder_name = parts[0]
            if len(parts) == 3:
                stage_name = parts[1]
                plate_name = parts[2]
            else:
                stage_name = None
                plate_name = parts[1]
        else:
            folder_name, plate_name, stage_name = image_name, None, None

        image_dir = (
            self.output_dir
            / experiment_name
            / folder_name
        )

        if not image_dir.exists():
            return None, None

        if plate_name == "Full (Watermarked)":
            wm_path = image_dir / f"{folder_name}_watermarked.jpg"
            orig_path = image_dir / f"{folder_name}_original.jpg"
            if wm_path.exists():
                return orig_path if orig_path.exists() else wm_path, wm_path
            else:
                return None, None

        original = None
        processed = None

        if stage_name:
            # Dapatkan base name dari plate
            # misalnya plate_name = "plate_00_segmented" atau "plate_00"
            plate_base = plate_name.replace("_preprocessed", "").replace("_enhanced", "").replace("_segmented", "")
            
            # Map untuk mencari Before image berdasarkan current stage
            if stage_name == "detection":
                orig_path = image_dir / f"{folder_name}_original.jpg"
                if orig_path.exists():
                    original = orig_path
                    
                processed_path = image_dir / "00_detection" / f"{plate_name}.jpg"
                if processed_path.exists():
                    processed = processed_path
                    
            elif stage_name == "preprocessing":
                # Before: Detection (crop)
                orig_path = image_dir / "00_detection" / f"{plate_base}.jpg"
                if orig_path.exists():
                    original = orig_path
                    
                processed_path = image_dir / "01_preprocessing" / f"{plate_name}.jpg"
                if processed_path.exists():
                    processed = processed_path
                    
            elif stage_name == "enhancement":
                # Before: Preprocessing
                orig_path = image_dir / "01_preprocessing" / f"{plate_base}_preprocessed.jpg"
                if orig_path.exists():
                    original = orig_path
                    
                processed_path = image_dir / "02_enhancement" / f"{plate_name}.jpg"
                if processed_path.exists():
                    processed = processed_path
                    
            elif stage_name == "segmentation":
                # Before: Enhancement
                orig_path = image_dir / "02_enhancement" / f"{plate_base}_enhanced.jpg"
                if orig_path.exists():
                    original = orig_path
                    
                processed_path = image_dir / "03_segmentation" / f"{plate_name}.jpg"
                if processed_path.exists():
                    processed = processed_path

        if processed is None:
            processed = original

        # Jika original tidak ditemukan (misal karena cache belum dihapus), fallback ke processed
        if original is None:
            original = processed

        return original, processed

    # =========================================================
    # FIND ALL IMAGES
    # =========================================================

    def _find_images(
        self,
        directory
    ):

        return sorted(
            [
                path
                for path in directory.rglob("*")
                if (
                    path.is_file()
                    and path.suffix.lower()
                    in self.IMAGE_EXTENSIONS
                )
            ]
        )


class ImageCanvasFrame(ttk.Frame):
    def __init__(self, parent, title):
        super().__init__(parent)
        self.zoom_factor = 1.0
        self.pil_image = None
        self.photo_image = None

        # Title
        ttk.Label(self, text=title, font=("Arial", 12, "bold")).pack(pady=(0,5))
        
        # Buttons
        btn_frame = ttk.Frame(self)
        btn_frame.pack(fill="x", pady=(0, 5))
        ttk.Button(btn_frame, text="Zoom In (+)", command=self.zoom_in).pack(side="left", expand=True, fill="x", padx=2)
        ttk.Button(btn_frame, text="Zoom Out (-)", command=self.zoom_out).pack(side="left", expand=True, fill="x", padx=2)
        ttk.Button(btn_frame, text="Reset", command=self.zoom_reset).pack(side="left", expand=True, fill="x", padx=2)

        # Canvas with Scrollbars
        canvas_frame = ttk.Frame(self)
        canvas_frame.pack(fill="both", expand=True)
        
        self.canvas = tk.Canvas(canvas_frame, bg="gray", cursor="cross")
        self.scroll_y = ttk.Scrollbar(canvas_frame, orient="vertical", command=self.canvas.yview)
        self.scroll_x = ttk.Scrollbar(canvas_frame, orient="horizontal", command=self.canvas.xview)
        
        self.canvas.configure(yscrollcommand=self.scroll_y.set, xscrollcommand=self.scroll_x.set)
        
        self.scroll_y.pack(side="right", fill="y")
        self.scroll_x.pack(side="bottom", fill="x")
        self.canvas.pack(side="left", fill="both", expand=True)

        # Pan support
        self.canvas.bind("<ButtonPress-1>", self.start_pan)
        self.canvas.bind("<B1-Motion>", self.do_pan)
        
        # Mousewheel zoom
        self.canvas.bind("<MouseWheel>", self.on_mousewheel)
        self.canvas.bind("<Button-4>", self.on_mousewheel)
        self.canvas.bind("<Button-5>", self.on_mousewheel)

    def load_image(self, path):
        if path is None or not path.exists():
            self.canvas.create_text(200, 200, text="Gambar tidak ditemukan.", fill="red")
            return
            
        try:
            self.pil_image = Image.open(path).convert("RGB")
            # Set initial zoom so image fits in 450x500 box approx
            w, h = self.pil_image.size
            ratio = min(450 / max(1, w), 500 / max(1, h))
            self.zoom_factor = min(1.0, ratio)
            self.update_image()
        except Exception as e:
            self.canvas.create_text(200, 200, text=f"Error:\n{e}", fill="red")

    def zoom_in(self):
        self.zoom_factor *= 1.2
        self.update_image()
        
    def zoom_out(self):
        self.zoom_factor *= 0.8
        self.update_image()
        
    def zoom_reset(self):
        if self.pil_image:
            w, h = self.pil_image.size
            self.zoom_factor = min(450 / max(1, w), 500 / max(1, h))
            self.update_image()

    def on_mousewheel(self, event):
        if event.num == 5 or event.delta < 0:
            self.zoom_out()
        if event.num == 4 or event.delta > 0:
            self.zoom_in()

    def update_image(self):
        if not self.pil_image: return
        
        new_w = int(self.pil_image.width * self.zoom_factor)
        new_h = int(self.pil_image.height * self.zoom_factor)
        
        if new_w > 0 and new_h > 0:
            resized = self.pil_image.resize((new_w, new_h), Image.Resampling.LANCZOS)
            self.photo_image = ImageTk.PhotoImage(resized)
            self.canvas.delete("all")
            self.canvas.create_image(0, 0, anchor="nw", image=self.photo_image)
            self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def start_pan(self, event):
        self.canvas.scan_mark(event.x, event.y)

    def do_pan(self, event):
        self.canvas.scan_dragto(event.x, event.y, gain=1)


class PreviewWindow:

    def __init__(self, parent, original_path, processed_path, title="Processed Result"):
        self.window = tk.Toplevel(parent)
        self.window.title(title)
        
        self.window.update_idletasks()
        width = 1000
        height = 750
        x = (self.window.winfo_screenwidth() // 2) - (width // 2)
        y = (self.window.winfo_screenheight() // 2) - (height // 2)
        self.window.geometry(f"{width}x{height}+{x}+{y}")
        self.window.minsize(700, 500)

        self.original_path = Path(original_path) if original_path else None
        self.processed_path = Path(processed_path) if processed_path else None

        self._build_ui()

    def _build_ui(self):
        main = ttk.Frame(self.window, padding=10)
        main.pack(fill="both", expand=True)

        ttk.Label(main, text="HASIL PROSES", font=("Arial", 14, "bold")).pack(pady=(0, 10))

        images_frame = ttk.Frame(main)
        images_frame.pack(fill="both", expand=True)

        self.left_panel = ImageCanvasFrame(images_frame, "SEBELUM")
        self.left_panel.pack(side="left", fill="both", expand=True, padx=(0, 5))

        self.right_panel = ImageCanvasFrame(images_frame, "SESUDAH")
        self.right_panel.pack(side="left", fill="both", expand=True, padx=(5, 0))

        self.path_var = tk.StringVar(value="")
        ttk.Label(main, textvariable=self.path_var, wraplength=900).pack(fill="x", pady=(10, 0))

        self._load_images()

    def _load_images(self):
        self.left_panel.load_image(self.original_path)
        self.right_panel.load_image(self.processed_path)

        paths_text = ""
        if self.original_path:
            paths_text += f"Sebelum: {self.original_path.resolve()}\n"
        if self.processed_path:
            paths_text += f"Sesudah: {self.processed_path.resolve()}"
            
        self.path_var.set(paths_text)