import pathlib
from pathlib import Path
#import sys
from magicgui import widgets
import numpy as np
#from scipy.interpolate import griddata
#from scipy.spatial import KDTree
import napari
from napari.utils.notifications import show_info
import cc3d

#from napari_cool_tools_vol_proc._masking_tools_funcs import project_2d_mask
from batch_tools_utils import (
    #fill_nan_nearest_neighbor_2D,
    find_and_process_outliers,
    generate_circular_mask,
    generate_depth_map,
    #generate_height_map_from_3D_points,
    generate_maps,
    generate_thick_map,
    #get_points_centroid_along_axis,
    load_bits_labels,
    #mask_3D_points_with_2D_boolean_mask,
    mask_components_within_theshold_of_points_distribution,
    #outlier_correction,
)

# 1. Create a dummy text file to process
dummy_file_path = pathlib.Path("dummy.txt")
with open(dummy_file_path, "w") as f:
    f.write("This is the first line.\n")
    f.write("This is the second line.\n")

class FileProcessorWidget(widgets.Container):
    """
    A custom widget with a file input and a button to process the file.
    """

    def __init__(self):
        super().__init__()

        # Create child widgets
        self.file_input = widgets.FileEdit(
            label="Select file:",
            mode="r",  # Set to "r" for reading an existing file
            tooltip="Select a file to process.",
            # value=Path(r"F:\38 peak stage ret_chor crop\UNPs_08874712-2023_09_06-14_02_11_processed_ret_chor_seg.npz") #dummy_file_path # Pre-fill with the dummy file path
            value=Path(
                r"E:\38 peak stage ret_chor crop\UNPs_08874712-2023_09_06-14_02_11_processed_ret_chor_seg.npz"
            ),  # dummy_file_path # Pre-fill with the dummy file path
        )
        self.process_button = widgets.PushButton(text="Run Function")

        # Add the widgets to the container
        self.append(self.file_input)
        self.append(self.process_button)

        # Connect the button's click event to a method
        self.process_button.changed.connect(self._on_button_clicked)

    def _on_button_clicked(self):
        """
        This method is called when the button is pressed.
        It runs the custom function with the current file path.
        """
        file_path = self.file_input.value
        label = load_bits_labels(file_path,"retina")
        #label = load_bits_labels(file_path, "choroid")
        raw_depth_map, ret_surf_coords, rpe_surf_coords, thick_map, difference_map = (
            generate_maps(label)
        )

        viewer.add_image(label, visible=False)
        viewer.add_image(raw_depth_map, visible=False)
        viewer.add_image(thick_map, visible=False)
        viewer.add_image(difference_map, visible=False)
        viewer.add_points(
            ret_surf_coords,
            size=4,
            face_color="red",
            name="retinal_surface",
            visible=False,
        )
        viewer.add_points(
            rpe_surf_coords,
            size=4,
            face_color="blue",
            name="rpe_surface",
            visible=False,
        )
        # viewer.add_points(retina_surf_centroids,size=4,face_color="magenta",name="retina_surf_centroids",visible=True)

        imaging_range = 12  # mm
        refractive_index = 1.33
        conv_factor = (
            imaging_range / label.shape[1] * 1000 / refractive_index
        )  # mm/pixel * um/mm /refractive index = um/pixel

        # threshold values
        gap_threshold = 8
        thickness_theshold = 500  # micrometers
        incedence_allowance = 1 / np.sin(np.pi / 4)
        # pixel_thickness_threshold = int(thickness_theshold / conv_factor)
        pixel_thickness_threshold = (
            int(thickness_theshold / conv_factor) * incedence_allowance
        )
        component_to_pixel_thickness_ratio = 1 / 3  # 1/2 #1/6 #1/8 #1/4 #1/2
        component_to_central_retina_threshold = (
            pixel_thickness_threshold * component_to_pixel_thickness_ratio
        )
        dust_threshold = 1e6
        print(
            f"Thresholdss\ngap: {gap_threshold}\nthickness theshold: {thickness_theshold}\nincedence allowance = {incedence_allowance}\npixel thicknes theshold: {pixel_thickness_threshold}\ndust threshold: {dust_threshold}\n"
        )
        print(
            f"component to pixel thickness ratio: {component_to_pixel_thickness_ratio}\ncomponent to central retina threshold: {component_to_central_retina_threshold}\n"
        )

        (
            clean,
            intersect2,
            recovered_components,
            intersect,
            diff2_3D,
            diff_3D,
            retina_surf_centroids,
            rpe_outlier_free,
            ret_outliers,
            ret_surf_replaced,
            ret_outlier_free,
            outlier_filled_height_map,
            outlier_map,
            diff2,
            diff,
        ) = find_and_process_outliers(
            label,
            raw_depth_map,
            difference_map,
            ret_surf_coords,
            rpe_surf_coords,
            gap_threshold=gap_threshold,
            pixel_thickness_threshold=pixel_thickness_threshold,
            component_to_central_retina_threshold=component_to_central_retina_threshold,
        )

        viewer.add_labels(diff, name="diff", visible=False)
        viewer.add_labels(diff2, name="diff2", visible=False)
        viewer.add_points(
            ret_surf_replaced,
            size=4,
            face_color="green",
            border_color="yellow",
            name="ret_surf_replaced",
            visible=True,
        )
        viewer.add_points(
            ret_outliers, size=4, face_color="red", border_color="orange", visible=True
        )
        viewer.add_points(
            ret_outlier_free,
            size=4,
            face_color="green",
            border_color="blue",
            visible=True,
        )
        viewer.add_labels(diff_3D, visible=False)
        viewer.add_labels(diff2_3D, visible=False)
        viewer.add_labels(intersect, visible=False)
        viewer.add_labels(recovered_components, visible=True)
        viewer.add_labels(intersect2, visible=False)
        viewer.add_image(outlier_filled_height_map, visible=False)
        viewer.add_labels(clean, visible=False)

        # process small remnants
        show_info("Processing Small Components\n")
        dust_free = cc3d.dust(clean, threshold=dust_threshold).astype("bool") * 6
        outliers = (label.astype("bool") & ~dust_free.astype("bool")) * 10
        outliers_coord_mask = (
            outliers.sum(axis=1).astype(bool) * 1
            - recovered_components.sum(axis=1).astype(bool) * 1
        ).astype(bool)
        num_outliers = np.count_nonzero(outliers)
        num_labeled = np.count_nonzero(label)
        show_info(
            f"{num_outliers}/{num_labeled} outliers {(num_outliers / num_labeled) * 100}%\n"
        )

        viewer.add_labels(dust_free, visible=True)
        viewer.add_labels(outliers)
        viewer.add_image(outliers_coord_mask)

        # TODO revisit this intersect would need to be recalculated to get an accurate reading with the recovered data
        # TODO verify that outlier_coord_mask is accurate and also recalculate. Then perform thickness calculations!!
        show_info("Processing Final Cleanup\n")
        (
            raw_depth_map2,
            ret_surf_coords2,
            rpe_surf_coords2,
            thick_map2,
            difference_map2,
        ) = generate_maps(dust_free.astype("bool") * 1)

        (
            clean2,
            intersect2_2,
            recovered_components2,
            intersect_2,
            diff2_3D2,
            diff_3D2,
            retina_surf_centroids2,
            rpe_outlier_free2,
            ret_outliers2,
            ret_surf_replaced2,
            ret_outlier_free2,
            outlier_filled_height_map2,
            outlier_map2,
            diff2_2,
            diff_2,
        ) = find_and_process_outliers(
            dust_free,
            raw_depth_map2,
            difference_map2,
            ret_surf_coords2,
            rpe_surf_coords2,
            gap_threshold=gap_threshold,
            pixel_thickness_threshold=pixel_thickness_threshold,
            component_to_central_retina_threshold=component_to_central_retina_threshold,
        )

        recovered_components2 = mask_components_within_theshold_of_points_distribution(
            intersect_2,
            point_distirbution=retina_surf_centroids2,
            threshold=component_to_central_retina_threshold,
        )

        squeaky_clean = (clean2.astype(bool) + recovered_components2) * 36

        outliers2 = (dust_free.astype("bool") & ~squeaky_clean.astype("bool")) * 10
        num_outliers2 = np.count_nonzero(outliers2)
        num_labeled2 = np.count_nonzero(dust_free)
        show_info(
            f"{num_outliers2}/{num_labeled2} outliers {(num_outliers2 / num_labeled2) * 100}%\n"
        )

        viewer.add_image(raw_depth_map2, visible=False)
        viewer.add_image(thick_map2, visible=False)
        viewer.add_image(difference_map2, visible=False)
        viewer.add_image(outlier_filled_height_map2, visible=False)
        viewer.add_labels(diff_3D2, visible=False)
        viewer.add_labels(diff2_3D2, visible=False)
        viewer.add_labels(outliers2)
        viewer.add_labels(recovered_components2, visible=True)
        viewer.add_labels(squeaky_clean)
        viewer.add_points(
            ret_surf_coords2,
            size=4,
            face_color="yellow",
            name="retinal_surface2",
            visible=True,
        )
        viewer.add_points(
            rpe_surf_coords2,
            size=4,
            face_color="purple",
            name="rpe_surface2",
            visible=True,
        )
        viewer.add_points(
            retina_surf_centroids2,
            size=4,
            face_color="magenta",
            name="retina_surf_centroids",
            visible=True,
        )
        viewer.add_points(
            ret_surf_replaced2,
            size=4,
            face_color="green",
            border_color="yellow",
            name="ret_surf_replaced",
            visible=True,
        )

if __name__ == "__main__":
    try:
        viewer = napari.Viewer()
        # Create and show the custom widget
        processor_widget = FileProcessorWidget()
        viewer.window.add_dock_widget(processor_widget)
        viewer.show()
        napari.run()
        # processor_widget.show(run=True)
    finally:
        # Clean up the dummy file when the widget is closed
        if dummy_file_path.exists():
            dummy_file_path.unlink()
