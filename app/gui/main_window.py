import tkinter as tk
from tkinter import ttk, messagebox

from app.pipeline.experiment_service import (
    ExperimentService
)

from app.gui.experiment_panel import (
    ExperimentPanel
)

from app.gui.result_panel import (
    ResultPanel
)

from app.gui.preview import (
    PreviewResolver,
    PreviewWindow
)


class MainWindow:

    def __init__(
        self,
        root
    ):

        self.root = root

        self.root.title(
            "License Plate Image Processing"
        )

        self.root.geometry(
            "1250x800"
        )

        self.root.minsize(
            1000,
            650
        )

        # =====================================================
        # SERVICE
        # =====================================================

        self.service = ExperimentService(
            output_dir="output"
        )

        # =====================================================
        # PREVIEW
        # =====================================================

        self.preview_resolver = PreviewResolver(
            output_dir="output"
        )

        # =====================================================
        # BUILD
        # =====================================================

        self._build_ui()

    # =========================================================
    # BUILD UI
    # =========================================================

    def _build_ui(self):

        container = ttk.Frame(
            self.root,
            padding=10
        )

        container.pack(
            fill="both",
            expand=True
        )

        # =====================================================
        # LEFT PANEL
        # =====================================================

        self.experiment_panel = (
            ExperimentPanel(
                container,
                on_run=self._run_experiment
            )
        )

        self.experiment_panel.pack(
            side="left",
            fill="y",
            padx=(0, 10)
        )

        # =====================================================
        # RIGHT PANEL
        # =====================================================

        right = ttk.Frame(
            container
        )

        right.pack(
            side="left",
            fill="both",
            expand=True
        )

        self.result_panel = (
            ResultPanel(
                right,
                service=self.service,
                on_preview=self._preview
            )
        )

        self.result_panel.pack(
            fill="both",
            expand=True
        )

        # =====================================================
        # STATUS
        # =====================================================

        self.status_var = tk.StringVar(
            value="Ready"
        )

        ttk.Label(
            right,
            textvariable=self.status_var,
            relief="sunken",
            anchor="w"
        ).pack(
            fill="x",
            pady=(8, 0)
        )

    # =========================================================
    # RUN EXPERIMENT
    # =========================================================

    def _run_experiment(
        self,
        image_paths,
        experiment_config
    ):

        try:

            self.status_var.set(
                "Menjalankan experiment..."
            )

            self.root.update_idletasks()

            self.service.run(
                image_paths=image_paths,
                experiment_configs=[
                    experiment_config
                ]
            )

            # Refresh hasil
            self.result_panel.refresh()

            self.status_var.set(
                "Experiment selesai."
            )

            messagebox.showinfo(
                "Success",
                "Experiment berhasil dijalankan."
            )

        except Exception as error:

            self.status_var.set(
                "Error"
            )

            messagebox.showerror(
                "Experiment Error",
                str(error)
            )

    # =========================================================
    # PREVIEW RESULT
    # =========================================================

    def _preview(
        self,
        experiment_name,
        image_name
    ):

        try:

            # -------------------------------------------------
            # Cari HASIL PROSES
            # -------------------------------------------------

            original_path, processed_path = (
                self.preview_resolver
                .find_original_and_processed(
                    experiment_name,
                    image_name
                )
            )

            # -------------------------------------------------
            # Tidak ditemukan
            # -------------------------------------------------

            if original_path is None and processed_path is None:

                messagebox.showerror(
                    "Preview",
                    (
                        "Hasil proses tidak ditemukan.\n\n"
                        f"Experiment : {experiment_name}\n"
                        f"Image      : {image_name}"
                    )
                )

                return

            # -------------------------------------------------
            # Buka hasil
            # -------------------------------------------------

            self.current_preview = PreviewWindow(
                self.root,
                original_path,
                processed_path,
                title=(
                    f"{experiment_name} - "
                    f"{image_name}"
                )
            )

            self.status_var.set(
                f"Menampilkan hasil: {image_name}"
            )

        except Exception as error:

            messagebox.showerror(
                "Preview Error",
                str(error)
            )


# =============================================================
# MAIN
# =============================================================

def main():

    root = tk.Tk()

    MainWindow(
        root
    )

    root.mainloop()


if __name__ == "__main__":
    main()