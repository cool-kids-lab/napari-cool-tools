import subprocess
import sys
from typing import List, Literal

import tomlkit
from magicgui import magicgui
import os

def main() -> bool:
    """Launches GUI to configure and install Napari with Pixi.

    Returns:
        True if installation succeeded, False otherwise.
    """

    @magicgui(call_button="Install")
    def napari_install(
        backend: Literal["cpu", "cuda12"] = "cpu"
    ):
        """Installs Napari with selected configuration.

        Args:
            version: Install the 'development' or 'production' build.
            backend: Backend type, e.g., 'cpu', 'cuda11', or 'cuda12'.
            visualization: Include visualization extras if True.
        """
        try:
            features: List[str] = [backend]

            # Load pixi.toml
            with open("pixi.toml", "r", encoding="utf-8") as file:
                print("File Opened")
                pixi_config = tomlkit.load(file)

            # Set features in the default environment
            pixi_config["environments"]["default"] = {"features": features}
            print(f"Default feature config set to: {features}\n")

            # Write updated config back
            with open("pixi.toml", "w", encoding="utf-8") as file:
                tomlkit.dump(pixi_config, file)

            # Run pixi install
            pixi_path = os.path.expanduser(r"~\.pixi\bin\pixi.exe")  # Typical path on Windows
            subprocess.check_call([pixi_path, "install"])

            print("Installation completed successfully.")
            sys.exit(0)

        except Exception as error:
            print(f"Installation failed: {error}")
            sys.exit(1)
    
    napari_install.native.setWindowTitle("Mode Selection")
    napari_install.show(run=True)
    return False  # Should never reach here unless GUI closes without action


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)