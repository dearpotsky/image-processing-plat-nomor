import tkinter as tk
from tkinter import ttk, filedialog

from app.pipeline.config import (
    PipelineConfig,
    EnhancementConfig,
    EnhancementStep,
    SegmentationConfig,
)

from app.pipeline.experiment import ExperimentConfig


class ExperimentPanel(ttk.LabelFrame):

    def __init__(self, parent, on_run=None):
        super().__init__(
            parent,
            text="Panel Proses",
            padding=10
        )

        self.on_run = on_run
        self.input_images = []
        self.enhancement_steps = []
        self.segmentation_steps = []
        # Detector padding variables (defaults)
        self.det_padding_v_var = tk.DoubleVar(value=0.12)
        self.det_padding_left_var = tk.DoubleVar(value=0.25)
        self.det_padding_right_var = tk.DoubleVar(value=0.20)
        # OCR parameters variables (defaults)
        self.ocr_text_thr_var = tk.DoubleVar(value=0.6)
        self.ocr_low_text_var = tk.DoubleVar(value=0.3)
        self.ocr_mag_ratio_var = tk.DoubleVar(value=1.2)

        self._build_ui()

    # ==================================================
    # UI
    # ==================================================

    def _build_ui(self):
        
        # Use columns to lay out horizontally and balance their widths
        col1 = ttk.Frame(self)
        col1.pack(side="left", fill="both", expand=True, padx=5)
        
        right_frame = ttk.Frame(self)
        right_frame.pack(side="left", fill="both", expand=True, padx=5)
        
        # The right frame holds col2 and col3 side-by-side
        cols_frame = ttk.Frame(right_frame)
        cols_frame.pack(side="top", fill="both", expand=True)
        
        col2 = ttk.Frame(cols_frame)
        col2.pack(side="left", fill="both", expand=True, padx=5)
        
        col3 = ttk.Frame(cols_frame)
        col3.pack(side="left", fill="both", expand=True, padx=5)

        # ----------------------------------------------
        # INPUT IMAGES & PROSES NAME (Col 1)
        # ----------------------------------------------
        
        ttk.Label(col1, text="Input Images").pack(anchor="w")

        self.input_listbox = tk.Listbox(col1, height=5, width=25)
        self.input_listbox.pack(fill="both", expand=False)

        button_frame = ttk.Frame(col1)
        button_frame.pack(fill="x", pady=(6, 0))

        ttk.Button(button_frame, text="Add", command=self._add_images).pack(side="left", fill="x", expand=True, padx=(0, 2))
        ttk.Button(button_frame, text="Remove", command=self._remove_images).pack(side="left", fill="x", expand=True, padx=(2, 0))

        ttk.Label(col1, text="Nama Proses").pack(anchor="w", pady=(10, 0))
        self.name_var = tk.StringVar(value="proses_01")
        ttk.Entry(col1, textvariable=self.name_var).pack(fill="x", pady=(3, 0))

        ttk.Label(col1, text="Watermark & OCR").pack(anchor="w", pady=(15, 0))
        self.watermark_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(col1, text="Aktifkan OCR Watermark", variable=self.watermark_var).pack(anchor="w")

        # ----------------------------------------------
        # ENHANCEMENT (Col 2)
        # ----------------------------------------------

        ttk.Label(col2, text="Tambah Enhancement").pack(anchor="w")
        self.enh_method_var = tk.StringVar()
        enhancement_methods = ["none", "contrast_stretching", "histogram_equalization", "lpf", "hpf", "retinex", "clahe", "gamma_correction", "unsharp_masking", "morphology_open"]
        self.enh_method_cb = ttk.Combobox(col2, textvariable=self.enh_method_var, values=enhancement_methods, state="readonly")
        self.enh_method_cb.current(0)
        self.enh_method_cb.pack(fill="x", pady=(3, 3))
        
        limit_frame = ttk.Frame(col2)
        limit_frame.pack(fill="x", pady=(0, 5))
        ttk.Label(limit_frame, text="Limit:").pack(side="left")
        self.enh_limit_var = tk.DoubleVar(value=0.0)
        ttk.Spinbox(limit_frame, from_=0.0, to=50.0, increment=0.1, textvariable=self.enh_limit_var, width=8).pack(side="left", padx=5)

        ttk.Button(col2, text="Tambahkan", command=self._add_enhancement_step).pack(fill="x")

        ttk.Label(col2, text="Urutan Enhancement").pack(anchor="w", pady=(5, 0))
        self.enhancement_listbox = tk.Listbox(col2, height=5, exportselection=False)
        self.enhancement_listbox.pack(fill="x", pady=(3, 3))
        
        ttk.Button(col2, text="Hapus", command=self._remove_enhancement_step).pack(fill="x", pady=(0, 10))

        # ----------------------------------------------
        # SEGMENTATION (Col 3)
        # ----------------------------------------------

        ttk.Label(col3, text="Tambah Segmentation").pack(anchor="w")
        self.seg_method_var = tk.StringVar()
        seg_methods = ["none", "global", "adaptive", "otsu", "sobel", "canny", "prewitt", "region_growing", "region_split_merge", "kmeans", "fuzzy_cmeans", "watershed", "active_contour", "clear_border"]
        self.seg_method_cb = ttk.Combobox(col3, textvariable=self.seg_method_var, values=seg_methods, state="readonly")
        self.seg_method_cb.current(0)
        self.seg_method_cb.pack(fill="x", pady=(3, 3))

        seg_limit_frame = ttk.Frame(col3)
        seg_limit_frame.pack(fill="x", pady=(0, 5))
        ttk.Label(seg_limit_frame, text="Limit:").pack(side="left")
        self.seg_limit_var = tk.DoubleVar(value=0.0)
        ttk.Spinbox(seg_limit_frame, from_=0.0, to=255.0, increment=1.0, textvariable=self.seg_limit_var, width=8).pack(side="left", padx=5)

        ttk.Button(col3, text="Tambahkan", command=self._add_segmentation_step).pack(fill="x")

        ttk.Label(col3, text="Urutan Segmentation").pack(anchor="w", pady=(5, 0))
        self.segmentation_listbox = tk.Listbox(col3, height=4, exportselection=False)
        self.segmentation_listbox.pack(fill="x", pady=(3, 3))

        ttk.Button(col3, text="Hapus", command=self._remove_segmentation_step).pack(fill="x")

        # ----------------------------------------------
        # RUN BUTTON (Spans across Col 2 and Col 3)
        # ----------------------------------------------
        ttk.Button(right_frame, text="PROSES SEKARANG", command=self._run).pack(side="bottom", fill="x", pady=(10,0))

        # Settings tabs (Detector & OCR)
        notebook = ttk.Notebook(self)
        notebook.pack(fill="both", expand=True, pady=(15,0))

        # Detector Settings tab
        det_frame = ttk.Frame(notebook)
        notebook.add(det_frame, text="Detector")
        ttk.Label(det_frame, text="Vertical Padding (0‑1)").grid(row=0, column=0, sticky="w", padx=5, pady=2)
        ttk.Spinbox(det_frame, from_=0.0, to=1.0, increment=0.01, textvariable=self.det_padding_v_var, width=8).grid(row=0, column=1, padx=5, pady=2)
        ttk.Label(det_frame, text="Left Padding (0‑1)").grid(row=1, column=0, sticky="w", padx=5, pady=2)
        ttk.Spinbox(det_frame, from_=0.0, to=1.0, increment=0.01, textvariable=self.det_padding_left_var, width=8).grid(row=1, column=1, padx=5, pady=2)
        ttk.Label(det_frame, text="Right Padding (0‑1)").grid(row=2, column=0, sticky="w", padx=5, pady=2)
        ttk.Spinbox(det_frame, from_=0.0, to=1.0, increment=0.01, textvariable=self.det_padding_right_var, width=8).grid(row=2, column=1, padx=5, pady=2)

        # OCR Settings tab
        ocr_frame = ttk.Frame(notebook)
        notebook.add(ocr_frame, text="OCR")
        ttk.Label(ocr_frame, text="Text Threshold (0‑1)").grid(row=0, column=0, sticky="w", padx=5, pady=2)
        ttk.Spinbox(ocr_frame, from_=0.0, to=1.0, increment=0.05, textvariable=self.ocr_text_thr_var, width=8).grid(row=0, column=1, padx=5, pady=2)
        ttk.Label(ocr_frame, text="Low Text (0‑1)").grid(row=1, column=0, sticky="w", padx=5, pady=2)
        ttk.Spinbox(ocr_frame, from_=0.0, to=1.0, increment=0.05, textvariable=self.ocr_low_text_var, width=8).grid(row=1, column=1, padx=5, pady=2)
        ttk.Label(ocr_frame, text="Mag Ratio (>0)").grid(row=2, column=0, sticky="w", padx=5, pady=2)
        ttk.Spinbox(ocr_frame, from_=0.1, to=5.0, increment=0.1, textvariable=self.ocr_mag_ratio_var, width=8).grid(row=2, column=1, padx=5, pady=2)

    # ==================================================
    # ADD IMAGES
    # ==================================================

    def _add_images(self):

        paths = filedialog.askopenfilenames(
            title="Select Images",
            filetypes=[
                (
                    "Image Files",
                    "*.jpg *.jpeg *.png *.bmp"
                ),
                (
                    "All Files",
                    "*.*"
                )
            ]
        )

        for path in paths:

            if path not in self.input_images:

                self.input_images.append(
                    path
                )

                self.input_listbox.insert(
                    tk.END,
                    path
                )

    # ==================================================
    # REMOVE IMAGES
    # ==================================================

    def _remove_images(self):

        selected = list(
            self.input_listbox.curselection()
        )

        for index in reversed(selected):

            self.input_listbox.delete(
                index
            )

            del self.input_images[
                index
            ]

    # ==================================================
    # ENHANCEMENT STEPS
    # ==================================================

    def _add_enhancement_step(self):
        method = self.enh_method_var.get()
        limit = self.enh_limit_var.get()
        
        if not method:
            return
            
        step = {"method": method, "limit": limit}
        self.enhancement_steps.append(step)
        self.enhancement_listbox.insert(tk.END, f"{method} (L: {limit})")
        
    def _remove_enhancement_step(self):
        selected = self.enhancement_listbox.curselection()
        if not selected:
            return
            
        selected_indices = sorted([int(i) for i in selected], reverse=True)
        for idx in selected_indices:
            self.enhancement_listbox.delete(idx)
            del self.enhancement_steps[idx]

    # ==================================================
    # SEGMENTATION STEPS
    # ==================================================

    def _add_segmentation_step(self):
        method = self.seg_method_var.get()
        limit = self.seg_limit_var.get()
        
        if not method:
            return
            
        step = {"method": method, "limit": limit}
        self.segmentation_steps.append(step)
        self.segmentation_listbox.insert(tk.END, f"{method} (L: {limit})")
        
    def _remove_segmentation_step(self):
        selected = self.segmentation_listbox.curselection()
        if not selected:
            return
            
        selected_indices = sorted([int(i) for i in selected], reverse=True)
        for idx in selected_indices:
            self.segmentation_listbox.delete(idx)
            del self.segmentation_steps[idx]

    # ==================================================
    # GET CONFIG
    # ==================================================

    def get_config(self):

        config = PipelineConfig()

        # Detector padding from UI
        config.detector.padding_v = self.det_padding_v_var.get()
        config.detector.padding_left = self.det_padding_left_var.get()
        config.detector.padding_right = self.det_padding_right_var.get()

        # OCR parameters from UI
        config.ocr.text_threshold = self.ocr_text_thr_var.get()
        config.ocr.low_text = self.ocr_low_text_var.get()
        config.ocr.mag_ratio = self.ocr_mag_ratio_var.get()

        steps = []
        for step_data in self.enhancement_steps:
            steps.append(EnhancementStep(
                method=step_data["method"],
                limit=step_data["limit"],
                tile_grid_width=8,
                tile_grid_height=8
            ))

        config.enhancement = EnhancementConfig(
            steps=steps
        )

        from app.pipeline.config import SegmentationStep
        seg_steps = []
        for step_data in self.segmentation_steps:
            seg_steps.append(SegmentationStep(
                method=step_data["method"],
                limit=step_data["limit"],
            ))
            
        if not seg_steps:
            seg_steps = [SegmentationStep(method="none")]

        config.segmentation = SegmentationConfig(
            steps=seg_steps
        )

        from app.pipeline.config import WatermarkConfig
        config.watermark = WatermarkConfig(
            enabled=self.watermark_var.get()
        )

        return config

    # ==================================================
    # RUN
    # ==================================================

    def _run(self):

        if not self.input_images:
            return

        name = self.name_var.get().strip()

        if not name:
            return

        experiment = ExperimentConfig(
            name=name,
            config=self.get_config()
        )

        if self.on_run:

            self.on_run(
                self.input_images.copy(),
                experiment
            )

    # ==================================================
    # PUBLIC
    # ==================================================

    def get_input_images(self):

        return self.input_images.copy()