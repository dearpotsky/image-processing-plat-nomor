import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from pathlib import Path
import logging

from app.pipeline.config import (
    PipelineConfig,
    EnhancementConfig,
    EnhancementStep,
    SegmentationConfig,
    SegmentationStep,
    WatermarkConfig,
)
from app.pipeline.config_utils import (
    save_config_json,
    load_config_json,
    save_last_config,
    load_last_config,
    LAST_CONFIG_PATH,
)
from app.pipeline.experiment import ExperimentConfig

logger = logging.getLogger("experiment_panel")


class ExperimentPanel(ttk.LabelFrame):

    def __init__(self, parent, on_run=None):
        super().__init__(
            parent,
            text="Panel Proses & Konfigurasi",
            padding=10
        )

        self.on_run = on_run
        self.input_images = []
        self.enhancement_steps = []
        self.segmentation_steps = []

        # ----------------------------------------------
        # Configuration Variables
        # ----------------------------------------------
        # Detector
        self.det_enabled_var = tk.BooleanVar(value=True)
        self.det_padding_v_var = tk.DoubleVar(value=0.12)
        self.det_padding_left_var = tk.DoubleVar(value=0.25)
        self.det_padding_right_var = tk.DoubleVar(value=0.20)

        # OCR
        self.ocr_engine_var = tk.StringVar(value="easyocr")
        self.ocr_text_thr_var = tk.DoubleVar(value=0.6)
        self.ocr_low_text_var = tk.DoubleVar(value=0.3)
        self.ocr_mag_ratio_var = tk.DoubleVar(value=1.2)

        # Preprocessing
        self.prep_resize_var = tk.BooleanVar(value=True)
        self.prep_grayscale_var = tk.BooleanVar(value=True)
        self.prep_normalize_var = tk.BooleanVar(value=True)

        # General
        self.watermark_var = tk.BooleanVar(value=True)
        self.name_var = tk.StringVar(value="proses_01")
        self.config_name_var = tk.StringVar(value="Default")

        self._build_ui()

    # ==================================================
    # UI
    # ==================================================

    def _build_ui(self):
        # ----------------------------------------------
        # TOOLBAR ATAS: Manajemen Konfigurasi (Save/Load)
        # ----------------------------------------------
        cfg_toolbar = ttk.Frame(self)
        cfg_toolbar.pack(side="top", fill="x", pady=(0, 10))

        ttk.Label(
            cfg_toolbar,
            text="Preset Konfigurasi:",
            font=('Segoe UI', 9, 'bold')
        ).pack(side="left", padx=(0, 6))

        ttk.Button(
            cfg_toolbar,
            text="💾 Simpan Konfig...",
            command=self._save_config_dialog,
            padding=3
        ).pack(side="left", padx=2)

        ttk.Button(
            cfg_toolbar,
            text="📂 Muat Konfig...",
            command=self._load_config_dialog,
            padding=3
        ).pack(side="left", padx=2)

        ttk.Button(
            cfg_toolbar,
            text="📌 Simpan sbg Default",
            command=self._save_as_default,
            padding=3
        ).pack(side="left", padx=2)

        ttk.Button(
            cfg_toolbar,
            text="🔄 Reset Bawaan",
            command=self._reset_to_default,
            padding=3
        ).pack(side="left", padx=2)

        self.cfg_status_label = ttk.Label(
            cfg_toolbar,
            text="[Aktif: Default]",
            foreground="#555",
            font=('Segoe UI', 9, 'italic')
        )
        self.cfg_status_label.pack(side="left", padx=(8, 0))

        # ----------------------------------------------
        # KOLOM PROSES UTAMA (Col 1, Col 2, Col 3)
        # ----------------------------------------------
        main_cols = ttk.Frame(self)
        main_cols.pack(side="top", fill="both", expand=True)

        col1 = ttk.Frame(main_cols)
        col1.pack(side="left", fill="both", expand=True, padx=5)

        right_frame = ttk.Frame(main_cols)
        right_frame.pack(side="left", fill="both", expand=True, padx=5)

        cols_frame = ttk.Frame(right_frame)
        cols_frame.pack(side="top", fill="both", expand=True)

        col2 = ttk.Frame(cols_frame)
        col2.pack(side="left", fill="both", expand=True, padx=5)

        col3 = ttk.Frame(cols_frame)
        col3.pack(side="left", fill="both", expand=True, padx=5)

        # ----------------------------------------------
        # INPUT IMAGES & PROSES NAME (Col 1)
        # ----------------------------------------------
        ttk.Label(col1, text="Input Images", font=('Segoe UI', 9, 'bold')).pack(anchor="w")

        self.input_listbox = tk.Listbox(col1, height=5, width=25)
        self.input_listbox.pack(fill="both", expand=False)

        btn_frame = ttk.Frame(col1)
        btn_frame.pack(fill="x", pady=(4, 0))

        ttk.Button(btn_frame, text="Add", command=self._add_images).pack(side="left", fill="x", expand=True, padx=(0, 2))
        ttk.Button(btn_frame, text="Remove", command=self._remove_images).pack(side="left", fill="x", expand=True, padx=(2, 0))

        ttk.Label(col1, text="Nama Proses", font=('Segoe UI', 9, 'bold')).pack(anchor="w", pady=(8, 0))
        ttk.Entry(col1, textvariable=self.name_var).pack(fill="x", pady=(2, 0))

        ttk.Label(col1, text="Watermark & Output", font=('Segoe UI', 9, 'bold')).pack(anchor="w", pady=(8, 0))
        ttk.Checkbutton(col1, text="Aktifkan OCR Watermark", variable=self.watermark_var).pack(anchor="w")

        # ----------------------------------------------
        # ENHANCEMENT (Col 2)
        # ----------------------------------------------
        ttk.Label(col2, text="Tambah Enhancement", font=('Segoe UI', 9, 'bold')).pack(anchor="w")
        self.enh_method_var = tk.StringVar()
        enhancement_methods = [
            "none", "contrast_stretching", "histogram_equalization", "lpf", "hpf",
            "retinex", "clahe", "gamma_correction", "unsharp_masking", "morphology_open"
        ]
        self.enh_method_cb = ttk.Combobox(
            col2,
            textvariable=self.enh_method_var,
            values=enhancement_methods,
            state="readonly"
        )
        self.enh_method_cb.current(0)
        self.enh_method_cb.pack(fill="x", pady=(2, 2))

        limit_frame = ttk.Frame(col2)
        limit_frame.pack(fill="x", pady=(0, 4))
        ttk.Label(limit_frame, text="Limit:").pack(side="left")
        self.enh_limit_var = tk.DoubleVar(value=0.0)
        ttk.Spinbox(
            limit_frame,
            from_=0.0,
            to=50.0,
            increment=0.1,
            textvariable=self.enh_limit_var,
            width=8
        ).pack(side="left", padx=5)

        ttk.Button(col2, text="Tambahkan", command=self._add_enhancement_step).pack(fill="x")

        ttk.Label(col2, text="Urutan Enhancement:").pack(anchor="w", pady=(4, 0))
        self.enhancement_listbox = tk.Listbox(col2, height=4, exportselection=False)
        self.enhancement_listbox.pack(fill="x", pady=(2, 2))

        ttk.Button(col2, text="Hapus", command=self._remove_enhancement_step).pack(fill="x", pady=(0, 6))

        # ----------------------------------------------
        # SEGMENTATION (Col 3)
        # ----------------------------------------------
        ttk.Label(col3, text="Tambah Segmentation", font=('Segoe UI', 9, 'bold')).pack(anchor="w")
        self.seg_method_var = tk.StringVar()
        seg_methods = [
            "none", "global", "adaptive", "otsu", "sobel", "canny", "prewitt",
            "region_growing", "region_split_merge", "kmeans", "fuzzy_cmeans",
            "watershed", "active_contour", "clear_border"
        ]
        self.seg_method_cb = ttk.Combobox(
            col3,
            textvariable=self.seg_method_var,
            values=seg_methods,
            state="readonly"
        )
        self.seg_method_cb.current(0)
        self.seg_method_cb.pack(fill="x", pady=(2, 2))

        seg_limit_frame = ttk.Frame(col3)
        seg_limit_frame.pack(fill="x", pady=(0, 4))
        ttk.Label(seg_limit_frame, text="Limit:").pack(side="left")
        self.seg_limit_var = tk.DoubleVar(value=0.0)
        ttk.Spinbox(
            seg_limit_frame,
            from_=0.0,
            to=255.0,
            increment=1.0,
            textvariable=self.seg_limit_var,
            width=8
        ).pack(side="left", padx=5)

        ttk.Button(col3, text="Tambahkan", command=self._add_segmentation_step).pack(fill="x")

        ttk.Label(col3, text="Urutan Segmentation:").pack(anchor="w", pady=(4, 0))
        self.segmentation_listbox = tk.Listbox(col3, height=4, exportselection=False)
        self.segmentation_listbox.pack(fill="x", pady=(2, 2))

        ttk.Button(col3, text="Hapus", command=self._remove_segmentation_step).pack(fill="x", pady=(0, 6))

        # ----------------------------------------------
        # TOMBOL PROSES SEKARANG
        # ----------------------------------------------
        ttk.Button(
            right_frame,
            text="🚀 PROSES SEKARANG",
            command=self._run
        ).pack(side="bottom", fill="x", pady=(8, 0))

        # ----------------------------------------------
        # TABS PENGATURAN (Detector, OCR, Preprocessing)
        # ----------------------------------------------
        notebook = ttk.Notebook(self)
        notebook.pack(fill="both", expand=True, pady=(10, 0))

        # --- TAB 1: DETECTOR ---
        det_frame = ttk.Frame(notebook, padding=8)
        notebook.add(det_frame, text="Deteksi Plat (YOLO)")

        ttk.Checkbutton(
            det_frame,
            text="Aktifkan Deteksi Plat YOLO (Matikan jika gambar input sudah berupa plat terpotong / close-up)",
            variable=self.det_enabled_var,
            command=self._on_det_toggle
        ).grid(row=0, column=0, columnspan=3, sticky="w", pady=(0, 5))

        self.det_mode_info = ttk.Label(
            det_frame,
            text="Mode Aktif: YOLO mendeteksi & memotong area plat kendaraan.",
            foreground="#27ae60",
            font=('Segoe UI', 9)
        )
        self.det_mode_info.grid(row=1, column=0, columnspan=3, sticky="w", pady=(0, 6))

        ttk.Label(det_frame, text="Vertical Padding (0‑1):").grid(row=2, column=0, sticky="w", padx=5, pady=2)
        self.spin_pad_v = ttk.Spinbox(det_frame, from_=0.0, to=1.0, increment=0.01, textvariable=self.det_padding_v_var, width=8)
        self.spin_pad_v.grid(row=2, column=1, sticky="w", padx=5, pady=2)

        ttk.Label(det_frame, text="Left Padding (0‑1):").grid(row=3, column=0, sticky="w", padx=5, pady=2)
        self.spin_pad_left = ttk.Spinbox(det_frame, from_=0.0, to=1.0, increment=0.01, textvariable=self.det_padding_left_var, width=8)
        self.spin_pad_left.grid(row=3, column=1, sticky="w", padx=5, pady=2)

        ttk.Label(det_frame, text="Right Padding (0‑1):").grid(row=4, column=0, sticky="w", padx=5, pady=2)
        self.spin_pad_right = ttk.Spinbox(det_frame, from_=0.0, to=1.0, increment=0.01, textvariable=self.det_padding_right_var, width=8)
        self.spin_pad_right.grid(row=4, column=1, sticky="w", padx=5, pady=2)

        # --- TAB 2: OCR ---
        ocr_frame = ttk.Frame(notebook, padding=8)
        notebook.add(ocr_frame, text="OCR (EasyOCR / PaddleOCR)")

        ttk.Label(ocr_frame, text="Engine:").grid(row=0, column=0, sticky="w", padx=5, pady=2)
        ocr_engines = ["easyocr", "paddleocr"]
        ttk.Combobox(
            ocr_frame,
            textvariable=self.ocr_engine_var,
            values=ocr_engines,
            state="readonly",
            width=14,
        ).grid(row=0, column=1, sticky="w", padx=5, pady=2)

        ttk.Label(ocr_frame, text="Text Threshold (0‑1):").grid(row=1, column=0, sticky="w", padx=5, pady=2)
        ttk.Spinbox(ocr_frame, from_=0.0, to=1.0, increment=0.05, textvariable=self.ocr_text_thr_var, width=8).grid(row=1, column=1, sticky="w", padx=5, pady=2)

        ttk.Label(ocr_frame, text="Low Text (0‑1):").grid(row=2, column=0, sticky="w", padx=5, pady=2)
        ttk.Spinbox(ocr_frame, from_=0.0, to=1.0, increment=0.05, textvariable=self.ocr_low_text_var, width=8).grid(row=2, column=1, sticky="w", padx=5, pady=2)

        ttk.Label(ocr_frame, text="Mag Ratio (>0):").grid(row=3, column=0, sticky="w", padx=5, pady=2)
        ttk.Spinbox(ocr_frame, from_=0.1, to=5.0, increment=0.1, textvariable=self.ocr_mag_ratio_var, width=8).grid(row=3, column=1, sticky="w", padx=5, pady=2)

        # --- TAB 3: PREPROCESSING ---
        prep_frame = ttk.Frame(notebook, padding=8)
        notebook.add(prep_frame, text="Preprocessing & Kualitas")

        ttk.Checkbutton(
            prep_frame,
            text="Aktifkan Resize Citra Plat (Matikan jika ingin menjaga ketajaman resolusi asli agar tidak pecah)",
            variable=self.prep_resize_var
        ).grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 4))

        ttk.Checkbutton(
            prep_frame,
            text="Ubah ke Grayscale",
            variable=self.prep_grayscale_var
        ).grid(row=1, column=0, columnspan=2, sticky="w", pady=(0, 4))

        ttk.Checkbutton(
            prep_frame,
            text="Normalisasi Kontras Min-Max (0-255)",
            variable=self.prep_normalize_var
        ).grid(row=2, column=0, columnspan=2, sticky="w", pady=(0, 4))

        self._on_det_toggle()

    def _on_det_toggle(self):
        """Update tampilan label info & disable padding saat deteksi nonaktif."""
        if self.det_enabled_var.get():
            self.det_mode_info.configure(
                text="✓ Mode Deteksi YOLO Aktif: mendeteksi dan memotong area plat kendaraan.",
                foreground="#27ae60"
            )
            state = "normal"
        else:
            self.det_mode_info.configure(
                text="⚡ Mode Direct Plate Aktif: Gambar langsung diproses utuh sebagai plat (tanpa YOLO crop, kualitas asli terjaga!).",
                foreground="#d35400"
            )
            state = "disabled"

        self.spin_pad_v.configure(state=state)
        self.spin_pad_left.configure(state=state)
        self.spin_pad_right.configure(state=state)

    # ==================================================
    # SAVE / LOAD CONFIGURATION
    # ==================================================

    def _save_config_dialog(self):
        """Menyimpan konfigurasi saat ini ke file JSON."""
        configs_dir = Path("configs")
        configs_dir.mkdir(exist_ok=True)

        file_path = filedialog.asksaveasfilename(
            initialdir=str(configs_dir),
            title="Simpan Konfigurasi Preset",
            defaultextension=".json",
            filetypes=[("JSON Files", "*.json"), ("All Files", "*.*")]
        )
        if not file_path:
            return

        config = self.get_config()
        try:
            save_config_json(config, file_path)
            preset_name = Path(file_path).stem
            self.cfg_status_label.configure(text=f"[Aktif: {preset_name}]", foreground="#2980b9")
            messagebox.showinfo("Berhasil", f"Konfigurasi berhasil disimpan ke:\n{file_path}")
        except Exception as e:
            messagebox.showerror("Error", f"Gagal menyimpan konfigurasi: {e}")

    def _load_config_dialog(self):
        """Memuat konfigurasi dari file JSON pilihan pengguna."""
        configs_dir = Path("configs")
        configs_dir.mkdir(exist_ok=True)

        file_path = filedialog.askopenfilename(
            initialdir=str(configs_dir),
            title="Muat Konfigurasi Preset",
            filetypes=[("JSON Files", "*.json"), ("All Files", "*.*")]
        )
        if not file_path:
            return

        try:
            config = load_config_json(file_path)
            self.set_config(config)
            preset_name = Path(file_path).stem
            self.cfg_status_label.configure(text=f"[Aktif: {preset_name}]", foreground="#2980b9")
            messagebox.showinfo("Berhasil", f"Konfigurasi '{preset_name}' berhasil dimuat!")
        except Exception as e:
            messagebox.showerror("Error", f"Gagal memuat konfigurasi: {e}")

    def _save_as_default(self):
        """Menjadikan konfigurasi saat ini sebagai default bawaan aplikasi."""
        config = self.get_config()
        save_last_config(config)
        self.cfg_status_label.configure(text="[Aktif: Default]", foreground="#27ae60")
        messagebox.showinfo(
            "Default Disimpan",
            "Konfigurasi saat ini telah disimpan sebagai default.\n"
            "Setting ini akan otomatis dimuat setiap kali aplikasi dibuka."
        )

    def _reset_to_default(self):
        """Mengembalikan konfigurasi ke nilai default standar sistem."""
        default_cfg = PipelineConfig()
        self.set_config(default_cfg)
        self.cfg_status_label.configure(text="[Aktif: Standar]", foreground="#555")

    # ==================================================
    # SET & GET CONFIG
    # ==================================================

    def set_config(self, config: PipelineConfig):
        """Memperbarui seluruh komponen GUI berdasarkan objek PipelineConfig."""
        if not config:
            return

        # Detector
        self.det_enabled_var.set(getattr(config.detector, "enabled", True))
        self.det_padding_v_var.set(config.detector.padding_v)
        self.det_padding_left_var.set(config.detector.padding_left)
        self.det_padding_right_var.set(config.detector.padding_right)

        # OCR
        self.ocr_engine_var.set(config.ocr.engine)
        self.ocr_text_thr_var.set(config.ocr.text_threshold)
        self.ocr_low_text_var.set(config.ocr.low_text)
        self.ocr_mag_ratio_var.set(config.ocr.mag_ratio)

        # Preprocessing
        self.prep_resize_var.set(getattr(config.preprocessing, "resize", True))
        self.prep_grayscale_var.set(config.preprocessing.grayscale)
        self.prep_normalize_var.set(config.preprocessing.normalize)

        # General
        self.watermark_var.set(config.watermark.enabled)
        if config.experiment and config.experiment.name:
            self.name_var.set(config.experiment.name)

        # Enhancement listbox
        self.enhancement_steps.clear()
        self.enhancement_listbox.delete(0, tk.END)
        if config.enhancement and config.enhancement.enabled:
            for step in config.enhancement.steps:
                if step.method and step.method != "none":
                    self.enhancement_steps.append({"method": step.method, "limit": step.limit})
                    self.enhancement_listbox.insert(tk.END, f"{step.method} (L: {step.limit})")

        # Segmentation listbox
        self.segmentation_steps.clear()
        self.segmentation_listbox.delete(0, tk.END)
        if config.segmentation and config.segmentation.enabled:
            for step in config.segmentation.steps:
                if step.method and step.method != "none":
                    self.segmentation_steps.append({"method": step.method, "limit": step.limit})
                    self.segmentation_listbox.insert(tk.END, f"{step.method} (L: {step.limit})")

        self._on_det_toggle()

    def get_config(self) -> PipelineConfig:
        """Mengambil setting dari seluruh kontrol UI dan membentuk PipelineConfig."""
        config = PipelineConfig()

        # Detector
        config.detector.enabled = self.det_enabled_var.get()
        config.detector.padding_v = self.det_padding_v_var.get()
        config.detector.padding_left = self.det_padding_left_var.get()
        config.detector.padding_right = self.det_padding_right_var.get()

        # OCR
        config.ocr.engine = self.ocr_engine_var.get()
        config.ocr.text_threshold = self.ocr_text_thr_var.get()
        config.ocr.low_text = self.ocr_low_text_var.get()
        config.ocr.mag_ratio = self.ocr_mag_ratio_var.get()

        # Preprocessing
        config.preprocessing.resize = self.prep_resize_var.get()
        config.preprocessing.grayscale = self.prep_grayscale_var.get()
        config.preprocessing.normalize = self.prep_normalize_var.get()

        # Enhancement
        enh_steps = []
        for step_data in self.enhancement_steps:
            enh_steps.append(EnhancementStep(
                method=step_data["method"],
                limit=step_data["limit"],
                tile_grid_width=8,
                tile_grid_height=8
            ))
        config.enhancement = EnhancementConfig(
            enabled=bool(enh_steps),
            steps=enh_steps
        )

        # Segmentation
        seg_steps = []
        for step_data in self.segmentation_steps:
            seg_steps.append(SegmentationStep(
                method=step_data["method"],
                limit=step_data["limit"],
            ))
        if not seg_steps:
            seg_steps = [SegmentationStep(method="none")]

        config.segmentation = SegmentationConfig(
            enabled=True,
            steps=seg_steps
        )

        # Watermark
        config.watermark = WatermarkConfig(
            enabled=self.watermark_var.get()
        )

        # Experiment
        config.experiment.name = self.name_var.get().strip() or "proses_01"

        return config

    # ==================================================
    # ADD & REMOVE IMAGES
    # ==================================================

    def _add_images(self):
        paths = filedialog.askopenfilenames(
            title="Pilih Gambar",
            filetypes=[
                ("Image Files", "*.jpg *.jpeg *.png *.bmp"),
                ("All Files", "*.*")
            ]
        )
        for path in paths:
            if path not in self.input_images:
                self.input_images.append(path)
                self.input_listbox.insert(tk.END, path)

    def _remove_images(self):
        selected = list(self.input_listbox.curselection())
        for index in reversed(selected):
            self.input_listbox.delete(index)
            del self.input_images[index]

    # ==================================================
    # ENHANCEMENT STEPS
    # ==================================================

    def _add_enhancement_step(self):
        method = self.enh_method_var.get()
        limit = self.enh_limit_var.get()
        if not method or method == "none":
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
        if not method or method == "none":
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
    # RUN
    # ==================================================

    def _run(self):
        if not self.input_images:
            messagebox.showwarning("Peringatan", "Harap pilih minimal satu gambar input terlebih dahulu.")
            return

        name = self.name_var.get().strip()
        if not name:
            name = "proses_01"

        config = self.get_config()

        # Otomatis simpan konfigurasi saat ini agar tidak hilang
        save_last_config(config)

        experiment = ExperimentConfig(
            name=name,
            config=config
        )

        if self.on_run:
            self.on_run(
                self.input_images.copy(),
                experiment
            )

    def get_input_images(self):
        return self.input_images.copy()