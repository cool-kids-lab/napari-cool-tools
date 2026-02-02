import napari
import numpy as np
from pathlib import Path
from magicgui import magicgui
from skimage.io import imread

# Initialize napari viewer
viewer = napari.Viewer()

# State for the image generator
state = {
    "image_gen": None,
    "current_path": None,
    "supported_exts": {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}
}

@magicgui(
    call_button=False,
    layout="vertical",
    directory={"widget_type": "FileEdit", "mode": "d", "label": "Select Directory"},
    next_btn={"widget_type": "PushButton", "text": "Next Image"},
    finetune_btn={"widget_type": "PushButton", "text": "Finetune"},
    inference_btn={"widget_type": "PushButton", "text": "Inference"},
)
def image_control_widget(
    directory=Path.home(),
    next_btn=None,
    finetune_btn=None,
    inference_btn=None
):
    pass

def load_next():
    """Loads the next valid image from the generator and updates layers."""
    if state["image_gen"] is None:
        return

    try:
        while True:
            img_path = next(state["image_gen"])
            if img_path.suffix.lower() in state["supported_exts"]:
                data = imread(str(img_path))
                
                # Update or create Image Layer
                if "Base Image" in viewer.layers:
                    viewer.layers["Base Image"].data = data
                else:
                    viewer.add_image(data, name="Base Image")
                
                # Update or create uint8 Labels Layer (Annotation Layer)
                # Labels layer handles integer values 0-255 natively
                mask_shape = data.shape[:2] if data.ndim == 3 else data.shape
                if "Annotations" in viewer.layers:
                    viewer.layers["Annotations"].data = np.zeros(mask_shape, dtype=np.uint8)
                else:
                    viewer.add_labels(np.zeros(mask_shape, dtype=np.uint8), name="Annotations")
                
                viewer.reset_view()
                print(f"Loaded: {img_path.name}")
                break
    except StopIteration:
        print("End of directory reached.")

# Connect Widget Actions
@image_control_widget.directory.changed.connect
def on_directory_changed(new_dir):
    state["image_gen"] = Path(new_dir).rglob("*")
    load_next()

@image_control_widget.next_btn.clicked.connect
def on_next_clicked():
    load_next()

@image_control_widget.finetune_btn.clicked.connect
def on_finetune_clicked():
    print("Finetune function has been activated.")

@image_control_widget.inference_btn.clicked.connect
def on_inference_clicked():
    print("Inference function has been activated.")

# Add the custom widget to the napari window
viewer.window.add_dock_widget(image_control_widget, area="right", name="Image Controls")

if __name__ == "__main__":
    napari.run()
















# import napari
# import numpy as np
# import torch
# from pathlib import Path
# from skimage.io import imread
# from magicgui import magicgui
# from model import create_unet
# from engine import run_inference, run_training_step

# # --- Initialize System ---
# # --- Correct Order of Operations ---
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# # 1. Create Model
# model = create_unet()
# # 2. Move to Device FIRST
# model.to(DEVICE)
# # 3. Create Optimizer SECOND (so it tracks the GPU parameters)
# optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
# viewer = napari.Viewer()

# # File state tracking
# state = {"files": [], "current_idx": 0}

# def load_image_at_index(idx):
#     """Updates layers with static names for internal model consistency."""
#     if not state["files"]: return
#     file_path = state["files"][idx]
    
#     # Load as intensity (single channel)
#     img_data = imread(file_path, as_gray=True).astype(np.float32)
    
#     # 1. Update Image (Fixed name: "Image")
#     if "Image" in viewer.layers:
#         viewer.layers["Image"].data = img_data
#     else:
#         viewer.add_image(img_data, name="Image")
    
#     # 2. Update Labels (Fixed name: "Labels")
#     if "Labels" in viewer.layers:
#         viewer.layers["Labels"].data = np.zeros(img_data.shape, dtype=np.int32)
#     else:
#         viewer.add_labels(np.zeros(img_data.shape, dtype=np.int32), name="Labels")
    
#     # Update title for user context
#     viewer.title = f"HITL Workflow - {file_path.name}"
#     viewer.reset_view()

# @magicgui(
#     call_button="Select Folder",
#     directory={"mode": "d", "label": "Folder:"},
# )
# def dir_loader(directory: Path = Path(r"\\192.168.1.3\coolkid\Beth Roti\Ridge Height Output")): # Path type hint is MANDATORY
#     """Scan directory for intensity images and load the first one."""
#     # Common 2026 intensity formats
#     extensions = ("*.tif", "*.tiff", "*.png", "*.npy")
#     files = []
#     for ext in extensions:
#         files.extend(list(directory.glob(ext)))
    
#     state["files"] = sorted(files)
#     state["current_idx"] = 0
    
#     if state["files"]:
#         load_image_at_index(0)
#     else:
#         print(f"No valid images found in {directory}")

# @magicgui(call_button="Next Image (N)")
# def next_image(dummy=None): # magicgui buttons often need a placeholder arg
#     """Navigate to the next image in the sorted list."""
#     if not state["files"]:
#         print("No folder selected yet.")
#         return
#     state["current_idx"] = (state["current_idx"] + 1) % len(state["files"])
#     load_image_at_index(state["current_idx"])

# @magicgui(call_button="Predict (P)")
# def predict_widget():
#     if "Image" not in viewer.layers: return
#     mask = run_inference(model, viewer.layers["Image"].data, DEVICE)
#     viewer.layers["Labels"].data = mask.astype(np.int32)

# @magicgui(call_button="Fine-tune (T)")
# def train_widget():
#     if "Image" not in viewer.layers or "Labels" not in viewer.layers: return
#     img = viewer.layers["Image"].data
#     lbl = viewer.layers["Labels"].data
#     loss = run_training_step(model, optimizer, img, lbl, DEVICE)
#     print(f"Loss: {loss:.4f}")
#     predict_widget()

# # Shortcuts & UI Setup
# viewer.bind_key('n', lambda v: next_image())
# viewer.bind_key('p', lambda v: predict_widget())
# viewer.bind_key('t', lambda v: train_widget())

# viewer.window.add_dock_widget(dir_loader, area='right', name="1. Setup")
# viewer.window.add_dock_widget([next_image, predict_widget, train_widget], 
#                               area='right', name="2. Loop")

# if __name__ == "__main__":
#     napari.run()
