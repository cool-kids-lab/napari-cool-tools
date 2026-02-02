"""
"""

import logging
from pathlib import Path

from diff_nurbs import NURBSSurface
import napari
import numpy as np
import torch
import torch.optim as optim

from napari_cool_tools_io._npz_reader import npz_file_reader
from experimental_utils import extract_brightest_pixel_along_axis

logging.basicConfig(
    level=logging.INFO, # Capture INFO and above
    format='%(levelname)s: %(message)s'
)

def place_uniformly(knots: torch.Tensor, spline_degree: int) -> None:
    """Sets up clamped uniform knot vectors for consistent boundary evaluation."""
    # NURBS require pinning the first and last p+1 knots to 0 and 1
    num_knot_vals = len(knots[spline_degree:-spline_degree])
    knot_vals = torch.linspace(0, 1, num_knot_vals, device=knots.device)
    knots[:spline_degree] = 0
    knots[spline_degree:-spline_degree] = knot_vals
    knots[-spline_degree:] = 1

def fit_nurbs_to_surface_coords(
    coords: torch.Tensor,
    grid_size: tuple[int, int],
    num_ctrl: tuple[int, int] = (64, 64),
    degree: tuple[int, int] = (3, 3),
    num_iterations: int = 1000,
    lr: float = 0.05
) -> torch.Tensor:
    """
    Fits a NURBS surface to coordinates using the diff-nurbs evaluator.
    
    Args:
        coords: (N, 3) tensor of points.
        grid_size: (H, W) original grid resolution (e.g., 850, 850).
        num_ctrl: (U, V) control point density.
    """
    device = coords.device
    n_u, n_v = num_ctrl
    p, q = degree
    h, w = grid_size

    # 1. Setup Clamped Knot Vectors
    kv_u = torch.empty(n_u + p + 1, device=device)
    kv_v = torch.empty(n_v + q + 1, device=device)
    place_uniformly(kv_u, p)
    place_uniformly(kv_v, q)

    # 2. Instantiate Layer (Knots must be passed to __init__)
    nurbs_layer = NURBSSurface(n_u, n_v, p, q, knots_x=kv_u, knots_y=kv_v)

    # 3. Initialize Control Points
    # Extract scalar bounds for U and V domains
    min_pt, max_pt = coords.min(dim=0).values, coords.max(dim=0).values
    u_init = torch.linspace(min_pt[0].item(), max_pt[0].item(), n_u, device=device)
    v_init = torch.linspace(min_pt[1].item(), max_pt[1].item(), n_v, device=device)
    
    grid_u, grid_v = torch.meshgrid(u_init, v_init, indexing='ij')
    z_init = torch.full_like(grid_u, coords[:, 2].mean())
    
    # Homogenous points (1, U, V, 4) -> [x, y, z, weight] where weight=1.0
    init_ctrl = torch.stack([grid_u, grid_v, z_init, torch.ones_like(grid_u)], dim=-1).unsqueeze(0)
    ctrl_pts_homo = torch.nn.Parameter(init_ctrl)

    # 4. Optimization Setup
    optimizer = optim.Adam([ctrl_pts_homo], lr=lr)
    u_eval = torch.linspace(0, 1, h, device=device)
    v_eval = torch.linspace(0, 1, w, device=device)
    
    # Match evaluator output shape (1, H, W, 3)
    target = coords.view(1, h, w, 3)

    for i in range(num_iterations):
        optimizer.zero_grad()
        
        # CORRECT CALL: Pass only the control points and a tuple of evaluation grids
        # This satisfies the 2-positional-argument (plus self) signature
        pred_pts = nurbs_layer.evaluate(ctrl_pts_homo, (u_eval, v_eval))
        
        loss = torch.nn.functional.mse_loss(pred_pts, target)
        loss.backward()
        optimizer.step()

        if i % 200 == 0:
            print(f"Iteration {i:04d} | Loss: {loss.item():.6f}")

    return ctrl_pts_homo.detach().squeeze(0)[..., :3]

def test_function():

    logging.info("Started!!")
    # viewer = napari.Viewer(show=False)
    test_data_path = Path(r"\\192.168.1.3\coolkid\Beth Roti\Ridge Height Output\08818962-2023_07_05-14_31_10_structure.npz")

    test_data = npz_file_reader(test_data_path,return_layer=False)

    logging.info(test_data.shape)

    brightest_coords = extract_brightest_pixel_along_axis(test_data,axis=1)

    logging.info(brightest_coords.shape)

    control_point_grid = fit_nurbs_to_surface_coords(torch.as_tensor(brightest_coords,dtype=torch.float), grid_size=(840,800), num_ctrl=(20,20),degree=(3,3),num_iterations=1000,lr=0.05)

    print(test_data.shape,brightest_coords.shape,control_point_grid.shape)

test_function()
    