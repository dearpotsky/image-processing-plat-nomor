import tkinter as tk
from tkinter import ttk, filedialog

from app.pipeline.config import (
    PipelineConfig,
    EnhancementConfig,
    SegmentationConfig,
)

from app.pipeline.experiment import ExperimentConfig


class ExperimentPanel(ttk.LabelFrame):

    def __init__(self, parent, on_run=None):
        super().__init__(
            parent,
            text="Experiment",
            padding=10
        )

        self.on_run = on_run
        self.input_images = []

        self._build_ui()

    # ==================================================
    # UI
    # ==================================================

    def _build_ui(self):

        # ----------------------------------------------
        # INPUT IMAGES
        # ----------------------------------------------

        input_frame = ttk.LabelFrame(
            self,
            text="Input Images",
            padding=8
        )

        input_frame.pack(
            fill="x",
            pady=(0, 10)
        )

        self.input_listbox = tk.Listbox(
            input_frame,
            height=7,
            width=35
        )

        self.input_listbox.pack(
            fill="x"
        )

        button_frame = ttk.Frame(
            input_frame
        )

        button_frame.pack(
            fill="x",
            pady=(6, 0)
        )

        ttk.Button(
            button_frame,
            text="Add Images",
            command=self._add_images
        ).pack(
            side="left",
            fill="x",
            expand=True,
            padx=(0, 3)
        )

        ttk.Button(
            button_frame,
            text="Remove",
            command=self._remove_images
        ).pack(
            side="left",
            fill="x",
            expand=True,
            padx=(3, 0)
        )

        # ----------------------------------------------
        # EXPERIMENT NAME
        # ----------------------------------------------

        ttk.Label(
            self,
            text="Experiment Name"
        ).pack(
            anchor="w"
        )

        self.name_var = tk.StringVar(
            value="experiment_01"
        )

        ttk.Entry(
            self,
            textvariable=self.name_var
        ).pack(
            fill="x",
            pady=(3, 10)
        )

        # ----------------------------------------------
        # ENHANCEMENT
        # ----------------------------------------------

        ttk.Label(
            self,
            text="Enhancement"
        ).pack(
            anchor="w"
        )

        self.enhancement_var = tk.StringVar(
            value="clahe"
        )

        ttk.Combobox(
            self,
            textvariable=self.enhancement_var,
            values=[
                "none",
                "clahe"
            ],
            state="readonly"
        ).pack(
            fill="x",
            pady=(3, 8)
        )

        # ----------------------------------------------
        # CLAHE CLIP LIMIT
        # ----------------------------------------------

        ttk.Label(
            self,
            text="CLAHE Clip Limit"
        ).pack(
            anchor="w"
        )

        self.clip_limit_var = tk.DoubleVar(
            value=2.0
        )

        ttk.Spinbox(
            self,
            from_=0.1,
            to=10.0,
            increment=0.1,
            textvariable=self.clip_limit_var
        ).pack(
            fill="x",
            pady=(3, 8)
        )

        # ----------------------------------------------
        # SEGMENTATION
        # ----------------------------------------------

        ttk.Label(
            self,
            text="Segmentation"
        ).pack(
            anchor="w"
        )

        self.segmentation_var = tk.StringVar(
            value="otsu"
        )

        ttk.Combobox(
            self,
            textvariable=self.segmentation_var,
            values=[
                "none",
                "otsu",
                "adaptive"
            ],
            state="readonly"
        ).pack(
            fill="x",
            pady=(3, 10)
        )

        # ----------------------------------------------
        # RUN
        # ----------------------------------------------

        ttk.Button(
            self,
            text="RUN EXPERIMENT",
            command=self._run
        ).pack(
            fill="x"
        )

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
    # GET CONFIG
    # ==================================================

    def get_config(self):

        config = PipelineConfig()

        config.enhancement = EnhancementConfig(
            method=self.enhancement_var.get(),
            clip_limit=self.clip_limit_var.get(),
            tile_grid_width=8,
            tile_grid_height=8
        )

        config.segmentation = SegmentationConfig(
            method=self.segmentation_var.get()
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