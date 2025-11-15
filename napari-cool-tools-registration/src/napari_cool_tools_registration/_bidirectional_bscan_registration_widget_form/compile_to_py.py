import sys
import os
import subprocess
from qtpy.QtWidgets import QApplication

import re

def patch_ui_py(py_file: str):
    """Fix PyQt/PySide-generated .py files from .ui by patching enum syntax."""

    with open(py_file, 'r', encoding='utf-8') as f:
        code = f.read()

    # Step 1: Replace all '::' with '.' globally
    code = code.replace('::', '.')

    # Step 2: Fix invalid enum access like Qt.QFrame.Box → QFrame.Box
    # These classes are NOT part of QtCore.Qt or Qt, but direct classes in QtWidgets
    widget_classes = ['QFrame']
    for cls in widget_classes:
        code = re.sub(r'\bQtCore\.Qt\.(\w+)', r'QtWidgets.\1', code)

    # Step 3: Fix Qt enums like QtWidgets.Qt.Orientation.Horizontal → QtCore.Qt.Horizontal
    code = re.sub(r'\bQtWidgets\.Qt\.(Orientation|AlignmentFlag|ScrollBarPolicy|CheckState)\.(\w+)',
                  r'QtCore.Qt.\2', code)

    with open(py_file, 'w', encoding='utf-8') as f:
        f.write(code)

    print(f"[UI Patch] Patched enum syntax in: {py_file}")


def compile_ui(ui_file: str, py_file: str):
    """Compile a .ui file to a .py file using the Qt backend (PyQt5 or PySide6), respecting qtpy if used."""

    if not os.path.exists(ui_file):
        raise FileNotFoundError(f"UI file '{ui_file}' not found.")

    # Determine backend via qtpy if present
    qt_api = os.environ.get("QT_API", "").lower()
    backend = None

    if qt_api in {"pyqt5", "pyqt"}:
        backend = "pyqt5"
    elif qt_api == "pyside6":
        backend = "pyside6"
    elif not qt_api:
        # Try to infer from imports
        try:
            import PyQt5  # noqa
            backend = "pyqt5"
        except ImportError:
            try:
                import PySide6  # type: ignore # noqa
                backend = "pyside6"
            except ImportError:
                raise ImportError("Neither PyQt5 nor PySide6 is installed, and QT_API not set.")

    # Use backend-specific compile command
    if backend == "pyqt5":
        cmd = [sys.executable, "-m", "PyQt5.uic.pyuic", ui_file, "-o", py_file]
    elif backend == "pyside6":
        cmd = [sys.executable, "-m", "PySide6.scripts.uic", ui_file, "-o", py_file]
    else:
        raise RuntimeError(f"Unsupported backend: {backend}")

    print(f"[UI Compiler] Compiling {ui_file} -> {py_file} using {backend}")
    try:
        subprocess.run(cmd, check=True)
    except FileNotFoundError:
        raise FileNotFoundError(f"Could not run command: {' '.join(cmd)}\nMake sure {backend} is properly installed.")


if __name__ == "__main__":
    print(os.getcwd())
    compile_ui("bidirectional_bscan_registration_form.ui", "_bidirectional_bscan_registration_form.py")
    # patch_ui_py("_bidirectional_bscan_registration_form.py")

    app = QApplication(sys.argv)

    from _bidirectional_bscan_registration_widget import Bidirectional_Bscan_Registration_Widget

    widget = Bidirectional_Bscan_Registration_Widget()
    widget.show()
    sys.exit(app.exec())