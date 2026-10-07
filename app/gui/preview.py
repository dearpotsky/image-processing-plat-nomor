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
            folder_name, plate_name = image_name.split(" - ", 1)
        else:
            folder_name, plate_name = image_name, None

        image_dir = (
            self.output_dir
            / experiment_name
            / folder_name
        )

        if not image_dir.exists():

            return None, None

        # -----------------------------------------------------
        # Cari original (00_detection)
        # -----------------------------------------------------

        original = None
        det_dir = image_dir / "00_detection"
        if det_dir.exists():
            if plate_name:
                plate_path = det_dir / f"{plate_name}.jpg"
                if plate_path.exists():
                    original = plate_path
            else:
                images = self._find_images(det_dir)
                if images:
                    original = images[0]

        # -----------------------------------------------------
        # Cari processed
        # -----------------------------------------------------
        
        processed = None
        for stage in self.STAGES:
            if stage == "00_detection":
                continue

            stage_dir = (
                image_dir
                / stage
            )

            if not stage_dir.exists():
                continue

            if plate_name:
                # glob matches plate_name followed by anything (e.g. _segmented.jpg)
                matches = list(stage_dir.glob(f"{plate_name}*.jpg"))
                if matches:
                    processed = matches[0]
                    break
            else:
                images = self._find_images(
                    stage_dir
                )

                if images:
                    processed = images[0]
                    break
                
        if processed is None:
            processed = original

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


class PreviewWindow:

    def __init__(
        self,
        parent,
        original_path,
        processed_path,
        title="Processed Result"
    ):

        self.window = tk.Toplevel(
            parent
        )

        self.window.title(
            title
        )

        self.window.geometry(
            "1000x750"
        )

        self.window.minsize(
            700,
            500
        )

        self.original_photo = None
        self.processed_photo = None

        self.original_path = (
            Path(original_path)
            if original_path
            else None
        )
        
        self.processed_path = (
            Path(processed_path)
            if processed_path
            else None
        )

        self._build_ui()

    # =========================================================
    # BUILD UI
    # =========================================================

    def _build_ui(self):

        main = ttk.Frame(
            self.window,
            padding=10
        )

        main.pack(
            fill="both",
            expand=True
        )

        # -----------------------------------------------------
        # TITLE
        # -----------------------------------------------------

        ttk.Label(
            main,
            text="HASIL PROSES",
            font=(
                "Arial",
                14,
                "bold"
            )
        ).pack(
            pady=(0, 10)
        )

        # -----------------------------------------------------
        # IMAGES FRAME
        # -----------------------------------------------------

        images_frame = ttk.Frame(main)
        images_frame.pack(fill="both", expand=True)

        left_frame = ttk.Frame(images_frame)
        left_frame.pack(side="left", fill="both", expand=True)

        ttk.Label(left_frame, text="SEBELUM", font=("Arial", 12, "bold")).pack(pady=(0,5))

        self.original_label = ttk.Label(
            left_frame,
            anchor="center",
            relief="solid"
        )
        self.original_label.pack(
            fill="both",
            expand=True
        )
        
        right_frame = ttk.Frame(images_frame)
        right_frame.pack(side="left", fill="both", expand=True, padx=(10, 0))

        ttk.Label(right_frame, text="SESUDAH", font=("Arial", 12, "bold")).pack(pady=(0,5))

        self.processed_label = ttk.Label(
            right_frame,
            anchor="center",
            relief="solid"
        )
        self.processed_label.pack(
            fill="both",
            expand=True
        )

        # -----------------------------------------------------
        # PATH
        # -----------------------------------------------------

        self.path_var = tk.StringVar(
            value=""
        )

        ttk.Label(
            main,
            textvariable=self.path_var,
            wraplength=900
        ).pack(
            fill="x",
            pady=(10, 0)
        )

        # -----------------------------------------------------
        # LOAD
        # -----------------------------------------------------

        self._load_image()

    # =========================================================
    # LOAD IMAGE
    # =========================================================

    def _load_image(self):

        def load_to_label(path, label):
            if path is None or not path.exists():
                label.configure(text="Gambar tidak ditemukan.")
                return None
            try:
                img = Image.open(path).convert("RGB")
                img.thumbnail((450, 500), Image.Resampling.LANCZOS)
                photo = ImageTk.PhotoImage(img)
                label.configure(image=photo, text="")
                label.image = photo
                return photo
            except Exception as error:
                label.configure(text=f"Error:\n{error}")
                return None

        self.original_photo = load_to_label(self.original_path, self.original_label)
        self.processed_photo = load_to_label(self.processed_path, self.processed_label)

        paths_text = ""
        if self.original_path:
            paths_text += f"Sebelum: {self.original_path.resolve()}\n"
        if self.processed_path:
            paths_text += f"Sesudah: {self.processed_path.resolve()}"
            
        self.path_var.set(paths_text)