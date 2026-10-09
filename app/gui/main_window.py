import tkinter as tk
from tkinter import ttk, messagebox
import logging

class TextHandler(logging.Handler):
    def __init__(self, text_widget):
        super().__init__()
        self.text_widget = text_widget
        
    def emit(self, record):
        msg = self.format(record)
        def append():
            self.text_widget.configure(state='normal')
            self.text_widget.insert(tk.END, msg + '\n')
            self.text_widget.configure(state='disabled')
            self.text_widget.yview(tk.END)
        self.text_widget.after(0, append)

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
        
        # Style
        style = ttk.Style()
        if 'clam' in style.theme_names():
            style.theme_use('clam')
        style.configure(".", font=('Segoe UI', 10))
        style.configure("TLabelframe.Label", font=('Segoe UI', 11, 'bold'), foreground='#2c3e50')
        style.configure("TButton", font=('Segoe UI', 10, 'bold'), padding=6)
        
        container = ttk.Frame(
            self.root,
            padding=10
        )

        container.pack(
            fill="both",
            expand=True
        )

        # =====================================================
        # TOP PANEL (Settings in one row)
        # =====================================================

        self.experiment_panel = (
            ExperimentPanel(
                container,
                on_run=self._run_experiment
            )
        )

        self.experiment_panel.pack(
            side="top",
            fill="x",
            pady=(0, 10)
        )

        # =====================================================
        # STATUS (Pinned to Bottom)
        # =====================================================

        self.status_var = tk.StringVar(
            value="Ready"
        )

        ttk.Label(
            container,
            textvariable=self.status_var,
            relief="sunken",
            anchor="w",
            padding=5,
            font=('Segoe UI', 9, 'italic')
        ).pack(
            side="bottom",
            fill="x",
            pady=(8, 0)
        )

        # =====================================================
        # BOTTOM PANEL
        # =====================================================

        bottom = ttk.Frame(
            container
        )

        bottom.pack(
            side="top",
            fill="both",
            expand=True
        )

        self.result_panel = (
            ResultPanel(
                bottom,
                service=self.service,
                on_preview=self._preview
            )
        )

        self.result_panel.pack(
            side="left",
            fill="y",
            padx=(0, 10)
        )
        
        self.viewer_frame = ttk.LabelFrame(
            bottom,
            text="Riwayat Proses & Viewer",
            padding=10
        )
        self.viewer_frame.pack(
            side="left",
            fill="both",
            expand=True
        )

        header_frame = ttk.Frame(self.viewer_frame)
        header_frame.pack(fill="x", pady=(0, 5))

        def clear_log():
            self.history_text.configure(state='normal')
            self.history_text.delete('1.0', tk.END)
            self.history_text.configure(state='disabled')

        ttk.Button(
            header_frame,
            text="Bersihkan Log",
            command=clear_log,
            padding=2
        ).pack(side="right")
        
        self.history_text = tk.Text(
            self.viewer_frame,
            font=('Consolas', 10),
            bg='#1e1e1e',
            fg='#d4d4d4',
            height=10,
            state='disabled'
        )
        self.history_text.pack(fill="both", expand=True)

        # Setup Logger
        text_handler = TextHandler(self.history_text)
        text_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s', '%H:%M:%S'))
        logger = logging.getLogger('pipeline')
        logger.setLevel(logging.INFO)
        logger.addHandler(text_handler)



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