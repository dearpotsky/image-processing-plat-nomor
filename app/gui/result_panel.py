import tkinter as tk
from tkinter import ttk, messagebox


class ResultPanel(ttk.Frame):

    def __init__(
        self,
        parent,
        service,
        on_preview=None
    ):
        super().__init__(parent)

        self.service = service
        self.on_preview = on_preview

        self.selected_experiment = None

        self._build_ui()
        self.refresh()

    # =========================================================
    # BUILD UI
    # =========================================================

    def _build_ui(self):

        # =====================================================
        # EXPERIMENT PANEL
        # =====================================================

        experiment_frame = ttk.LabelFrame(
            self,
            text="Experiments",
            padding=8
        )

        experiment_frame.pack(
            fill="x",
            pady=(0, 10)
        )

        self.experiment_listbox = tk.Listbox(
            experiment_frame,
            height=6,
            exportselection=False
        )

        self.experiment_listbox.pack(
            fill="x"
        )

        self.experiment_listbox.bind(
            "<<ListboxSelect>>",
            self._on_experiment_selected
        )

        # Experiment buttons

        experiment_buttons = ttk.Frame(
            experiment_frame
        )

        experiment_buttons.pack(
            fill="x",
            pady=(6, 0)
        )

        ttk.Button(
            experiment_buttons,
            text="Refresh",
            command=self.refresh
        ).pack(
            side="left",
            fill="x",
            expand=True,
            padx=(0, 4)
        )

        ttk.Button(
            experiment_buttons,
            text="Delete Experiment",
            command=self.delete_experiment
        ).pack(
            side="left",
            fill="x",
            expand=True,
            padx=(4, 0)
        )

        # =====================================================
        # IMAGE RESULT PANEL
        # =====================================================

        image_frame = ttk.LabelFrame(
            self,
            text="Processed Images",
            padding=8
        )

        image_frame.pack(
            fill="both",
            expand=True
        )

        self.image_listbox = tk.Listbox(
            image_frame,
            height=10,
            exportselection=False
        )

        self.image_listbox.pack(
            fill="both",
            expand=True
        )

        # Double click = preview
        self.image_listbox.bind(
            "<Double-Button-1>",
            self._on_image_double_click
        )

        # Enter = preview
        self.image_listbox.bind(
            "<Return>",
            self._on_image_enter
        )

        # =====================================================
        # IMAGE BUTTONS
        # =====================================================

        image_buttons = ttk.Frame(
            image_frame
        )

        image_buttons.pack(
            fill="x",
            pady=(6, 0)
        )

        ttk.Button(
            image_buttons,
            text="Open Result",
            command=self.preview
        ).pack(
            side="left",
            fill="x",
            expand=True,
            padx=(0, 4)
        )

        ttk.Button(
            image_buttons,
            text="Delete Image",
            command=self.delete_image
        ).pack(
            side="left",
            fill="x",
            expand=True,
            padx=(4, 0)
        )

    # =========================================================
    # REFRESH EXPERIMENT
    # =========================================================

    def refresh(self):

        self.experiment_listbox.delete(
            0,
            tk.END
        )

        self.image_listbox.delete(
            0,
            tk.END
        )

        self.selected_experiment = None

        try:

            experiments = (
                self.service.list_experiments()
            )

        except Exception as error:

            messagebox.showerror(
                "Error",
                f"Gagal membaca experiment:\n\n{error}"
            )

            return

        for experiment in experiments:

            if isinstance(
                experiment,
                dict
            ):

                name = experiment.get(
                    "name",
                    ""
                )

            else:

                name = str(
                    experiment
                )

            if name:

                self.experiment_listbox.insert(
                    tk.END,
                    name
                )

    # =========================================================
    # EXPERIMENT SELECTED
    # =========================================================

    def _on_experiment_selected(
        self,
        event=None
    ):

        selection = (
            self.experiment_listbox.curselection()
        )

        if not selection:
            return

        experiment = (
            self.experiment_listbox.get(
                selection[0]
            )
        )

        self.selected_experiment = experiment

        # LANGSUNG LOAD IMAGE
        self._load_images(
            experiment
        )

    # =========================================================
    # LOAD IMAGES
    # =========================================================

    def _load_images(
        self,
        experiment
    ):

        self.image_listbox.delete(
            0,
            tk.END
        )

        try:

            images = self.service.list_images(
                experiment
            )

        except Exception as error:

            messagebox.showerror(
                "Error",
                f"Gagal membaca hasil image:\n\n{error}"
            )

            return

        for image in images:

            if isinstance(
                image,
                dict
            ):

                name = image.get(
                    "name",
                    ""
                )

            else:

                name = str(
                    image
                )

            if name:

                self.image_listbox.insert(
                    tk.END,
                    name
                )

    # =========================================================
    # GET SELECTED EXPERIMENT
    # =========================================================

    def get_selected_experiment(self):

        selection = (
            self.experiment_listbox.curselection()
        )

        if not selection:
            return None

        return self.experiment_listbox.get(
            selection[0]
        )

    # =========================================================
    # GET SELECTED IMAGE
    # =========================================================

    def get_selected_image(self):

        selection = (
            self.image_listbox.curselection()
        )

        if not selection:
            return None

        return self.image_listbox.get(
            selection[0]
        )

    # =========================================================
    # DOUBLE CLICK IMAGE
    # =========================================================

    def _on_image_double_click(
        self,
        event=None
    ):

        self.preview()

    # =========================================================
    # ENTER IMAGE
    # =========================================================

    def _on_image_enter(
        self,
        event=None
    ):

        self.preview()

    # =========================================================
    # PREVIEW RESULT
    # =========================================================

    def preview(self):

        experiment = (
            self.get_selected_experiment()
        )

        image = (
            self.get_selected_image()
        )

        if not experiment:

            messagebox.showwarning(
                "Open Result",
                "Pilih experiment terlebih dahulu."
            )

            return

        if not image:

            messagebox.showwarning(
                "Open Result",
                "Pilih image hasil proses."
            )

            return

        if self.on_preview is None:

            messagebox.showerror(
                "Open Result",
                "Preview handler belum terhubung."
            )

            return

        # HANYA KIRIM EXPERIMENT + IMAGE
        self.on_preview(
            experiment,
            image
        )

    # =========================================================
    # DELETE IMAGE
    # =========================================================

    def delete_image(self):

        experiment = (
            self.get_selected_experiment()
        )

        image = (
            self.get_selected_image()
        )

        if not experiment or not image:
            return

        confirm = messagebox.askyesno(
            "Delete Image",
            (
                "Hapus hasil proses?\n\n"
                f"Experiment : {experiment}\n"
                f"Image      : {image}"
            )
        )

        if not confirm:
            return

        try:

            self.service.delete_image(
                experiment,
                image
            )

            self._load_images(
                experiment
            )

        except Exception as error:

            messagebox.showerror(
                "Delete Error",
                str(error)
            )

    # =========================================================
    # DELETE EXPERIMENT
    # =========================================================

    def delete_experiment(self):

        experiment = (
            self.get_selected_experiment()
        )

        if not experiment:
            return

        confirm = messagebox.askyesno(
            "Delete Experiment",
            (
                "Hapus seluruh experiment?\n\n"
                f"{experiment}"
            )
        )

        if not confirm:
            return

        try:

            self.service.delete_experiment(
                experiment
            )

            self.refresh()

        except Exception as error:

            messagebox.showerror(
                "Delete Error",
                str(error)
            )