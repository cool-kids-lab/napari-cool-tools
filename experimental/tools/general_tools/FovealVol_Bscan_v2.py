import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import tkinter as tk
from tkinter import filedialog
import os.path as ospath
import xml.etree.ElementTree as ET
from pathlib import Path

from scipy.ndimage import map_coordinates
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


from skimage.transform import rotate
import polarTransform
#from skimage.transform import warp_polar

import matplotlib.gridspec as gridspec

data_element_size = 4  # number of bytes per data element f32 == 4 bytes

def ini_proc_word(line,target_str):
    """"""
    words = line.split("=")
    index = words.index(target_str)
    if index + 1 < len(words):
        val = int(words[index + 1])
        return val
    else:
        print("ERROR in ini_proc_word function")
        return None

def prof_get_reader(path):
    """Reader for COOL lab .prof file format.

    Args:
        path(str or list of str): Path to file, or list of paths.

    Returns:
        function or None
        If the path is a recognized format, return a function that accepts the
        same path or list of paths, and returns a list of layer data tuples.
    """
    # If format is recogized return reader function
    if isinstance(path, str) and path.endswith(".prof"):
        # calculate file size in bytes
        file_size = os.path.getsize(path)

        # calculate number of data entries
        # in this case we are using 32 bit floating point
        # aka 4 bytes  as there are 8 bits per byte
        num_entries = file_size / data_element_size

        meta = prof_proc_meta(path, ".prof")
        h, w, d, bmscan, w_param, dtype, layer_type = meta
        dot_prof = np.dtype(("<f4", (h, w)))
        volume = np.fromfile(path, dtype=dot_prof, count=-1)
        #volume = volume.transpose(0, 2, 1)
        # display = display[:,::-1,:]
        volume = np.flip(volume.transpose(0, 2, 1), 1)

        globals()["prof_width"] = w
        globals()["prof_height"] = h
        globals()["prof_depth"] = d
        globals()["prof_bmscan"] = bmscan
        globals()["prof_width_param"] = w_param
        globals()["dtype"] = dtype
        globals()["layer_type"] = layer_type

        return volume
    return None


def prof_proc_meta(path, ext: str):
    """Process .prof file metadata.

    Args:
        path(str or list of str): Path to file, or list of paths.
        ext(str): extension of source file

    Returns:
        If .ini metafile is valid returns tuple(height(int),width(int),depth(int),bmscan(int),width_param(int),dtype(None/dtype),layer_type(None/layer_type))
        else if .xml metafile is valid returns tuple(height(int),width(int),depth(int),bmscan(int),width_param(int),dtype(None/dtype),layer_type(None/layer_type))
        else returns None

        If both .ini and .xml metafiles exist the .ini file will be used and the .xml will be ingnored
    """

    height = None
    depth = None

    # print(f"\nOpening file: {path}")

    head, tail = ospath.split(path)

    # isolate file name from path and .prof extension
    # file_name = ospath.basename(path)
    file_name = tail

    # remove .prof extenstion
    file_no_ext = file_name.replace(ext, "")

    # remove common .prof specifiers _OCTA and _Struc
    file_base = file_no_ext.replace("_OCTA", "").replace("_Struc", "")

    # constuct path to metafile assumed to be in same directory
    meta_path = ospath.join(head, file_base + ".xml")
    # print(f"Associated .xml meta data file: {meta_path}")
    meta_path2 = ospath.join(head, file_base + ".ini")
    # print(f"Associated .ini meta data file: {meta_path2}")

    # verify whether meta file exists or not
    # if isinstance(meta_path, str):

    if Path(meta_path2).is_file():
        # print(".ini Meta Data exists:")
        width_param, height, width, depth, bmscan = (
            None,
            None,
            None,
            None,
            None,
        )

        data = {"section":"", "content":""}
        settings = []

        with open(meta_path2) as file:
            for i,line in enumerate(file):
                if "[" not in line:
                    data["content"] = f"{data['content']}{line}"

                    if data["section"] == "General" and "WIDTH=" in line:
                        width_param = ini_proc_word(line,"WIDTH")
                    if data["section"] == "General" and "HEIGHT=" in line:
                        height = ini_proc_word(line,"HEIGHT")
                    if data["section"] == "General" and "FRAMES=" in line:
                        depth = ini_proc_word(line,"FRAMES")
                    if data["section"] == "OCT" and "BScanWidth=" in line:
                        width = ini_proc_word(line,"BScanWidth")
                    if data["section"] == "OCTA" and "BMScan=" in line:
                        bmscan = ini_proc_word(line,"BMScan")
                else:
                    if i != 0:
                        settings.append(data)
                    data = {"section":"", "content":""}
                    data["section"] = line.replace("[","").replace("]","").replace("\n","")
                    settings
            settings.append(data)        

        dtype = None
        layer_type = None

        # Case no valid values obtained from metafile return None
        if (
            depth is not None
            and height is not None
            and width is not None
            and bmscan is not None
            # and width_param is not None
        ):
            return (
                height,
                width,
                depth,
                bmscan,
                width_param,
                dtype,
                layer_type,
            )
        else:
            return None

    if Path(meta_path).is_file():
        # print(".xml Meta Data exists:")

        tree = ET.parse(meta_path)
        root = tree.getroot()
        volume_size = root.find(".//Volume_Size")
        volume_size_attrib = volume_size.attrib
        if "Width" in volume_size_attrib:
            width_param = int(volume_size_attrib["Width"])
        else:
            width_param = None
        height = int(volume_size_attrib["Height"])
        width = int(volume_size_attrib["BscanWidth"])
        depth = int(volume_size_attrib["Number_of_Frames"])

        scanning_params = root.find(".//Scanning_Parameters")
        if scanning_params is not None:
            scanning_params_attrib = scanning_params.attrib
            bmscan = int(scanning_params_attrib["Number_of_BM_scans"])
        else:
            bmscan = None

        layer_info = root.find(".//Layer_Info")

        if layer_info is not None:
            layer_info_attrib = layer_info.attrib
            dtype = layer_info_attrib["Dtype"]
            layer_type = layer_info_attrib["Layer_Type"]
        else:
            dtype = None
            layer_type = None

        # Case no valid values obtained from metafile return None
        if (
            depth is not None
            and height is not None
            and width is not None
            # and bmscan is not None
            # and width_param is not None
        ):
            return (
                height,
                width,
                depth,
                bmscan,
                width_param,
                dtype,
                layer_type,
            )
        else:
            return None

    # case no metadata request path to metadata or cancel file load
    else:
        return None


def prof_file_reader(path):
    """Take a path or list of paths to .prof files and return a list of LayerData tuples.

    Args:
        path(str or list of str): Path to file, or list of paths.

    Returns:
        layer_data : list of tuples
            A list of LayerData tuples where each tuple in the list contains
            (data, metadata, layer_type), where data is a numpy array, metadata is
            a dict of keyword arguments for the corresponding viewer.add_* method
            in napari, and layer_type is a lower-case string naming the type of
            layer. Both "meta", and "layer_type" are optional. napari will
            default to layer_type=="image" if not provided
    """

    h = globals()["prof_height"]
    w = globals()["prof_width"]
    bmscan = globals()["prof_bmscan"]
    dtype = globals()["dtype"]
    layer_type = globals()["layer_type"]

    # isolate file name from path and .prof extension
    # file_name = ospath.basename(path)
    head, tail = ospath.split(path)
    file_name = tail.replace(".", "_")

    # define chuncks as little endian f32 4 byte floats with HEIGHT values
    # per row and WIDTH values per column
    if dtype is None:
        dot_prof = np.dtype(("<f4", (h, w)))
    else:
        dot_prof = np.dtype((dtype, (h, w)))

    # generate numpy array by loading 400 * 496 * f32 sized data chunks
    # and stacking them until end of file is reached
    b_scan = np.fromfile(path, dtype=dot_prof, count=-1)

    # transpose array so that x and y are switched then flip array
    # to better orient b-scans for manual segmentation
    display = b_scan.transpose(0, 2, 1)
    # display = display[:,::-1,:]
    display = np.flip(b_scan.transpose(0, 2, 1), 1)
    # display = b_scan

    # # Determine if volume is octa
    # if bmscan is not None and bmscan > 1:
    #     print("This is an OCTA Volume!!")
    #     sign = 1
    #     fix_octa = np.empty_like(display)
    #     for i in range(len(fix_octa)):
    #         if sign == -1:
    #             fix_octa[i] = np.flip(display[i], axis=1)
    #             #fix_octa[i] = display[i]
    #         else:
    #             fix_octa[i] = display[i]
    #         if (i + 1) % bmscan == 0:
    #             sign = sign * -1

    #     display = fix_octa

    # optional kwargs for viewer.add_* method
    add_kwargs = {"name": file_name}

    # optional layer type argument
    if layer_type is None:
        layer_type = "image"
    else:
        pass

    print(
        f"layer_name: {file_name}, shape: {display.shape}, dtype: {display.dtype}, layer type: {layer_type}\n" #bmscan: {bmscan},
    )
    return display

def cartify (slc):
    image = slc
    width, height = image.shape
 
    #crop = int(round(width*0.1))
    #image = rotate(image, 90, resize = True) #to turn theta axis to Y axis as needed for polarTransform
    #image = image[:,crop:width]
    height_pad = int(round(height*2.5)) #estimates for being in water and typical ref arm position
    image = np.pad(image, [(height_pad,0), (0,0)] , mode = 'constant', constant_values = 0)
    #image = np.pad(image, [(height_pad,0),(0,0)] , mode = 'constant', constant_values = 0)
    #print(image.shape)
    
    cart_image, ptSettings = polarTransform.convertToCartesianImage(image, initialAngle = -52*np.pi/180, finalAngle = 52*np.pi/180, hasColor = True)
    
    #cart_image = rotate(cart_image, -90, resize = True)
    #print(cart_image.shape)
    #cart_image = cart_image[:,:,:3]
    #print(cart_image.shape)
    #cart_image = np.clip(cart_image, 0, 1)
    height, width = cart_image.shape
    cart_image = cart_image[int(height/2):height, :]
    return cart_image, ptSettings

def line_to_edges(p1, p2, xdim, ydim):
    """
    Extend line through p1 and p2 until it hits image boundaries.
    Returns two points (x,y) on the boundary.
    """
    x1, y1 = p1
    x2, y2 = p2
    dx = x2 - x1
    dy = y2 - y1
    points = []

    if dx != 0:
        # Intersect with x=0
        t = (0 - x1) / dx
        y = y1 + t * dy
        if 0 <= y <= ydim - 1:
            points.append((0, y))

        # Intersect with x=xdim-1
        t = (xdim - 1 - x1) / dx
        y = y1 + t * dy
        if 0 <= y <= ydim - 1:
            points.append((xdim - 1, y))

    if dy != 0:
        # Intersect with y=0
        t = (0 - y1) / dy
        x = x1 + t * dx
        if 0 <= x <= xdim - 1:
            points.append((x, 0))

        # Intersect with y=ydim-1
        t = (ydim - 1 - y1) / dy
        x = x1 + t * dx
        if 0 <= x <= xdim - 1:
            points.append((x, ydim - 1))

    if len(points) >= 2:
        return points[0], points[1]
    else:
        raise RuntimeError("Line did not intersect with two edges properly.")

def extract_bscan_plane(volume, p1, p2, num_samples=None, order=1, slab_thickness=3):
    """
    Extract a 2D oblique B-scan plane (with optional slab averaging) from a 3D OCT volume.

    Parameters
    ----------
    volume : ndarray, shape (Y, Z, X)
        3D OCT data (Y=row, Z=depth, X=col).
    p1, p2 : tuple
        Start and end points (x, y) in the XY plane.
    num_samples : int, optional
        Number of samples along the oblique line. If None, use line length.
    order : int
        Interpolation order for map_coordinates.
    slab_thickness : int
        Number of pixels to average orthogonal to the line (>=1).

    Returns
    -------
    plane : ndarray, shape (Z, num_samples)
        Extracted oblique B-scan (averaged if slab_thickness > 1).
    """
    ydim, zdim, xdim = volume.shape

    # Extend the line to full image boundaries
    p1_ext, p2_ext = line_to_edges(p1, p2, xdim, ydim)
    y1, x1 = p1_ext
    y2, x2 = p2_ext

    # Length of line in XY
    length = int(np.hypot(x2 - x1, y2 - y1))
    if num_samples is None:
        num_samples = length

    # Sampling along main line
    x_line = np.linspace(x1, x2, num_samples)
    y_line = np.linspace(y1, y2, num_samples)

    # Perpendicular unit vector (for slab offsets)
    dx = x2 - x1
    dy = y2 - y1
    norm = np.hypot(dx, dy)
    if norm == 0:
        raise ValueError("p1 and p2 cannot be the same point")
    perp = np.array([-dy, dx]) / norm

    # Collect slabs
    slab_planes = []

    half = slab_thickness // 2
    offsets = range(-half, half + 1) if slab_thickness > 1 else [0]

    for o in offsets:
        x_offset = x_line + perp[0] * o
        y_offset = y_line + perp[1] * o

        # Build coordinate grid (Z vs line)
        z_vals = np.arange(zdim)
        zz, xx = np.meshgrid(z_vals, np.arange(num_samples), indexing="ij")

        coords = np.vstack([
            y_offset[xx.ravel()],   # Y
            zz.ravel(),             # Z
            x_offset[xx.ravel()]    # X
        ])

        plane = map_coordinates(volume, coords, order=order, mode="nearest")
        plane = plane.reshape(zdim, num_samples)
        slab_planes.append(plane)

    # Average across slab
    plane = np.mean(slab_planes, axis=0)
    cc_plane, _ = cartify(plane)
    #return plane

    return plane, cc_plane, (x1,y1), (x2,y2)

# === UTILITY FUNCTIONS ===
def flatten_data(data):
    return np.sum(data, axis=1)

def volume_calc(mask, center, radius):
    shape = mask.shape
    Y, X = np.ogrid[:shape[0], :shape[2]]
    dist_from_center = (X - center[1])**2 + (Y - center[0])**2
    circ_mask_2d = dist_from_center <= radius**2

    retina_mask = (mask == 1).astype(np.uint8)
    choroid_mask = (mask == 2).astype(np.uint8)

    retina_flat = np.sum(retina_mask, axis=1)
    choroid_flat = np.sum(choroid_mask, axis=1)

    vol_retina = np.sum(np.where(circ_mask_2d, retina_flat, 0))
    vol_choroid = np.sum(np.where(circ_mask_2d, choroid_flat, 0))
    return vol_retina, vol_choroid

def radial_profile(data, center):
    y, x = np.indices(data.shape)
    r = np.sqrt((x - center[1])**2 + (y - center[0])**2).astype(np.uint16)
    non_zero_mask = data > 0
    r = r[non_zero_mask]
    data = data[non_zero_mask]
    tbin = np.bincount(r.ravel(), data.ravel())
    nr = np.bincount(r.ravel())
    return tbin / nr

def profile_radial_smoother(array, smooth, window='flat'):
    s = np.r_[array[smooth-1:0:-1], array, array[-2:-smooth-1:-1]]
    if window == 'flat':
        w = np.ones(smooth, 'd')
    else:
        w = eval('np.' + window + '(smooth)')
    y = np.convolve(w / w.sum(), s, mode='valid')
    return y[:len(array)]

def dist_to(p1, p2):
    return np.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

# === POINT SELECTION GUI ===
def select_points(mask):
    img = np.sum(mask == 1, axis=1)
    picked_points = []
    labels = ['Fovea', 'Optic Nerve']
    skip_flag = {'skip': False}

    fig, ax = plt.subplots()
    ax.imshow(img, vmax=120)
    ax.set_title("Click Fovea, then Optic Nerve.\nPress Enter to confirm, 'r' to reset, 's' to skip.")

    point_artists = []

    def reset_points():
        nonlocal picked_points, point_artists
        picked_points = []
        for artist in point_artists:
            artist.remove()
        point_artists = []
        fig.canvas.draw()

    def onclick(event):
        if event.inaxes != ax or len(picked_points) >= 2:
            return
        if event.xdata is None or event.ydata is None:
            return
        x, y = int(event.xdata), int(event.ydata)
        picked_points.append((x, y))
        label = labels[len(picked_points) - 1]
        pt = ax.plot(x, y, 'ro')[0]
        txt = ax.text(x + 5, y, label, color='yellow', fontsize=12, weight='bold')
        point_artists.extend([pt, txt])
        fig.canvas.draw()

    def onkey(event):
        if event.key == 'enter' and len(picked_points) == 2:
            plt.close()
        elif event.key == 'r':
            reset_points()
        elif event.key == 's':
            skip_flag['skip'] = True
            plt.close()

    fig.canvas.mpl_connect('button_press_event', onclick)
    fig.canvas.mpl_connect('key_press_event', onkey)
    plt.show()

    if skip_flag['skip']:
        return None, None, True
    if len(picked_points) != 2:
        return None, None, False
    return picked_points[0][::-1], picked_points[1][::-1], False

# === MAIN BATCH PROCESS ===
def batch_process():
    root = tk.Tk()
    root.withdraw()
    filepath = filedialog.askopenfilename(
        title="Select mask .npy file",
        filetypes=[("NumPy files", "*.npy"), ("All files", "*.*")]
    )
    if not filepath:
        return
    filepath_volume = filedialog.askopenfilename(
        title="Select data file",
        filetypes=[("PROF", "*.prof"), ("All files", "*.*")]
    )

    results = []
    save_dir = os.path.dirname(filepath)
    distances_px = []

   
    file = os.path.basename(filepath)
    mask = np.load(filepath)
    data = prof_get_reader(filepath_volume)
    print(data.shape)
    #data = np.load(filepath_volume)

    print(f"\nProcessing {file}")
    fovea, optic_disc, skipped = select_points(mask)

    radius = dist_to(fovea, optic_disc)
    distances_px.append(radius)

    retina = (mask == 1).astype(np.uint8)
    choroid = (mask == 2).astype(np.uint8)
    flat_retina = flatten_data(retina)
    flat_choroid = flatten_data(choroid)

    prof_retina = profile_radial_smoother(radial_profile(flat_retina, fovea), smooth=3)
    prof_choroid = profile_radial_smoother(radial_profile(flat_choroid, fovea), smooth=3)

    vol_retina, vol_choroid = volume_calc(mask, fovea, radius)

    def safe_thick(profile, dist):
        return profile[dist] if dist < len(profile) else np.nan

    results.append({
        'filename': file,
        'fovea_y': fovea[0],
        'fovea_x': fovea[1],
        'optic_disc_y': optic_disc[0],
        'optic_disc_x': optic_disc[1],
        'vol_retina': vol_retina,
        'vol_choroid': vol_choroid,
        'thick_fovea': safe_thick(prof_retina, 0),
        'thick_25px': safe_thick(prof_retina, 25),
        'thick_50px': safe_thick(prof_retina, 50),
        'thick_75px': safe_thick(prof_retina, 75),
        'thick_100px': safe_thick(prof_retina, 100),
        'retina_profile': prof_retina,
        'choroid_profile': prof_choroid
    })

    #bscan = extract_bscan_plane(data, fovea, optic_disc, num_samples=None)
    ret_img = np.sum(mask == 1, axis=1)  # shape (Y, X)
    chor_img = np.sum(mask ==2, axis=1)
    # extract oblique plane (returns plane in (Z, samples) and and edge points in (x,y))
    bscan, cc_bscan, edge_p1_xy, edge_p2_xy = extract_bscan_plane(data, fovea, optic_disc,
                                                        num_samples=None,
                                                        order=1,
                                                        slab_thickness=5)
    #
    overlay_mask = True
    if overlay_mask == True:
        segments,cc_segments, edge_p1_xy, edge_p2_xy = extract_bscan_plane(mask, fovea, optic_disc,
                                                            num_samples=None,
                                                            order=1,
                                                            slab_thickness=5)
        # Define the colors for your values (0: transparent, 1: red, 2: blue)
        colors = [(0, 0, 0, 0),  # Transparent for value 0 (RGBA: R, G, B, Alpha)
                (1, 0, 0, 1),  # Red for value 1
                (0, 0, 1, 1)]  # Blue for value 2

        # Create a colormap from the list of colors
        cmap = mcolors.ListedColormap(colors)

        # Define the boundaries for your values
        # These boundaries define where each color in the colormap applies.
        # For values 0, 1, 2, we need 4 boundaries: -0.5, 0.5, 1.5, 2.5
        # This ensures that 0 falls between -0.5 and 0.5, 1 between 0.5 and 1.5, etc.
        bounds = [-0.5, 0.5, 1.5, 2.5]

        # Create a BoundaryNorm to map values to colors based on the defined boundaries
        norm = mcolors.BoundaryNorm(bounds, cmap.N)


    x_edge_coords = [edge_p1_xy[0], edge_p2_xy[0]]
    y_edge_coords = [edge_p1_xy[1], edge_p2_xy[1]]

    # Create a figure
    fig = plt.figure(figsize=(8, 6))

    gs = gridspec.GridSpec(2, 3, figure=fig)

    x_edge_coords = [edge_p1_xy[0], edge_p2_xy[0]]
    y_edge_coords = [edge_p1_xy[1], edge_p2_xy[1]]

    # Top-left subplot for the first square image
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(ret_img, vmax =120)
    ax1.set_title('Retinal Thickness')
    ax1.axis('off') # Hide axes for cleaner image display
    ax1.axis('off')
    ax1.plot(x_edge_coords, y_edge_coords, linestyle="--", color="grey", linewidth=1)
    ax1.scatter([fovea[1], optic_disc[1]], [fovea[0], optic_disc[0]], c='yellow', s=10, edgecolors='k')
    ax1.set_xlim([0, ret_img.shape[1] - 1])
    ax1.set_ylim([ret_img.shape[0] - 1, 0])  # invert y-axis so image displays normally
    ax1.axis('off')

    # Top-right subplot for the second square image
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(chor_img, vmax = 120)
    ax2.set_title('Choroid Thickness')
    ax2.axis('off')
    ax2.plot(x_edge_coords, y_edge_coords, linestyle="--", color="grey", linewidth=1)
    ax2.scatter([fovea[1], optic_disc[1]], [fovea[0], optic_disc[0]], c='yellow', s=10, edgecolors='k')
    ax2.set_xlim([0, chor_img.shape[1] - 1])
    ax2.set_ylim([chor_img.shape[0] - 1, 0])  # invert y-axis so image displays normally
    ax2.axis('off')

    ax3 = fig.add_subplot(gs[0,2]) # gs[1, :] means row 1, all columns
    ax3.imshow(bscan, cmap = "gray", vmin = 0.08, vmax = 0.4)
    ax3.set_title('B-scan')
    ax3.axis('off')

    ax4 = fig.add_subplot(gs[1, :]) # gs[2, :] means row 2, all columns
    ax4.imshow(cc_bscan, cmap = "gray", vmin = 0.08, vmax = 0.4)
    ax4.imshow(cc_segments, cmap = cmap, alpha = 0.3)
    ax4.set_title('Curve Corrected B-scan')
    ax4.axis('off')
    plt.tight_layout()
    plt.show()
    #plt.savefig(os.path.splitext(filepath)+ "_B-scan.png")

    if not results:
        print("No files processed.")
        return

    df = pd.DataFrame(results)
    # df.to_pickle(os.path.join(save_dir, "analysis_results.pkl"))
    # df.drop(columns=['retina_profile', 'choroid_profile']).to_excel(
    #     os.path.join(save_dir, "analysis_results.xlsx"), index=False
    # )

    print("\nProcessing complete.")

    print(f"Saved full data to analysis_results.pkl and Excel to analysis_results.xlsx in {save_dir}")

    # === FINAL AVERAGE RADIAL PROFILE PLOT ===
    min_length = 400
    retina_profiles = [r[:min_length] for r in df['retina_profile']]
    choroid_profiles = [c[:min_length] for c in df['choroid_profile']]

    def_range = 100

    retina_array = np.array(retina_profiles)

    max_vals_retina = []
    max_idxs_retina = []
    for ret_profile in retina_array:
        max_vals_retina.append(np.max(ret_profile[:def_range]))
        max_idxs_retina.append(np.argmax(ret_profile[:def_range]))

    choroid_array = np.array(choroid_profiles)

    max_vals_choroid = []
    max_idxs_choroid = []
    for chor_profile in choroid_array:
        max_vals_choroid.append(np.max(chor_profile[:def_range]))
        max_idxs_choroid.append(np.argmax(chor_profile[:def_range]))

    avg_retina = np.mean(retina_array, axis=0)
    avg_choroid = np.mean(choroid_array, axis=0)

    df['max_thickness_retina'] = max_vals_retina
    df['max_thickness_retina_distance'] = max_idxs_retina
    df['max_thickness_choroid'] = max_vals_choroid
    df['max_thickness_choroid_distance'] = max_idxs_choroid

    df.to_pickle(os.path.join(save_dir, "analysis_results.pkl"))
    df.drop(columns=['retina_profile', 'choroid_profile']).to_excel(
        os.path.join(save_dir, "analysis_results.xlsx"), index=False
    )

    sem_retina = np.std(retina_array, axis=0) / np.sqrt(retina_array.shape[0])
    sem_choroid = np.std(choroid_array, axis=0) / np.sqrt(choroid_array.shape[0])

    ci_retina = 1.96 * sem_retina
    ci_choroid = 1.96 * sem_choroid

    x = np.arange(min_length)
    

    plt.figure(figsize=(10, 6))
    plt.plot(x, avg_retina, color='blue', label="Average Retina Profile", linewidth=2)
    plt.fill_between(x, avg_retina - ci_retina, avg_retina + ci_retina, color='blue', alpha=0.3)

    plt.plot(x, avg_choroid, color='red', label="Average Choroid Profile", linewidth=2)
    plt.fill_between(x, avg_choroid - ci_choroid, avg_choroid + ci_choroid, color='red', alpha=0.3)

    plt.xlabel("Distance from Fovea (pixels)")
    plt.ylabel("Thickness (pixels)")
    plt.title("Average Radial Profiles with 95% CI")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # === PRINT AVERAGE DISTANCE IN PIXELS ===
    avg_distance_px = np.mean(distances_px)
    print(f"\nAverage distance (Fovea â†’ Optic Disc): {avg_distance_px:.2f} pixels")

# === RUN ===
if __name__ == "__main__":
    batch_process()