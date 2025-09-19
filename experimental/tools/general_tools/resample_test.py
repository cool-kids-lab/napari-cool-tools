"""Generated with the help of Gemini"""

import numpy as np

def get_full_line_coordinates(p1, p2, height, width):
    """
    Returns a list of all integer coordinates on the infinite line defined
    by p1 and p2 that are within the array's boundaries.

    Args:
        p1 (tuple): A point (row, col) on the line.
        p2 (tuple): Another point (row, col) on the line.
        height (int): The number of rows in the array.
        width (int): The number of columns in the array.

    Returns:
        list: A list of coordinate tuples (row, col).
    """
    # Define the array boundaries
    min_row, max_row = 0, height - 1
    min_col, max_col = 0, width - 1

    # Extract coordinates and handle vertical line case
    r1, c1 = p1
    r2, c2 = p2

    if c1 == c2:  # Vertical line
        line_coords = [(r, c1) for r in range(min(r1, r2), max(r1, r2) + 1)]
        # Add points to extend the line to the top and bottom edges
        if c1 >= min_col and c1 <= max_col:
            line_coords.extend([(r, c1) for r in range(min_row, max_row + 1)])
    else:  # General case for non-vertical lines
        # Determine endpoints of the line at the array's boundaries
        slope = (r2 - r1) / (c2 - c1)

        # Find intersections with the vertical boundaries (left and right)
        r_at_min_c = r1 + slope * (min_col - c1)
        r_at_max_c = r1 + slope * (max_col - c1)

        # Find intersections with the horizontal boundaries (top and bottom)
        c_at_min_r = c1 + (1 / slope) * (min_row - r1) if slope != 0 else float('inf')
        c_at_max_r = c1 + (1 / slope) * (max_row - r1) if slope != 0 else float('inf')

        # Collect potential boundary intersection points
        boundary_points = []
        # Left boundary
        if min_row <= r_at_min_c <= max_row:
            boundary_points.append((r_at_min_c, min_col))
        # Right boundary
        if min_row <= r_at_max_c <= max_row:
            boundary_points.append((r_at_max_c, max_col))
        # Top boundary
        if min_col <= c_at_min_r <= max_col:
            boundary_points.append((min_row, c_at_min_r))
        # Bottom boundary
        if min_col <= c_at_max_r <= max_col:
            boundary_points.append((max_row, c_at_max_r))

        # Ensure we have at least two boundary points to define the extended line
        if len(boundary_points) < 2:
            return []  # Line does not intersect the array

        # Find the two distinct points to use as the new endpoints
        # These are the furthest two points on the line that are inside the array
        start_point = tuple(map(int, boundary_points[0]))
        end_point = tuple(map(int, boundary_points[1]))

        # Use Bresenham's algorithm on the new, extended endpoints
        # This implementation is adapted to work in all directions.
        line_coords = []
        r_start, c_start = start_point
        r_end, c_end = end_point

        dr = abs(r_end - r_start)
        dc = abs(c_end - c_start)

        r, c = r_start, c_start
        sr = -1 if r_start > r_end else 1
        sc = -1 if c_start > c_end else 1

        if dr > dc:
            err = dr / 2.0
            while r != r_end:
                err -= dc
                if err < 0:
                    c += sc
                    err += dr
                r += sr
                line_coords.append((r, c))
        else:
            err = dc / 2.0
            while c != c_end:
                err -= dr
                if err < 0:
                    r += sr
                    err += dc
                c += sc
                line_coords.append((r, c))
        line_coords.append(end_point)

    # Filter out points that are outside the array bounds,
    # as floating-point calculations can sometimes lead to points
    # just outside the integer range.
    final_coords = []
    for r, c in line_coords:
        if min_row <= r <= max_row and min_col <= c <= max_col:
            final_coords.append((r, c))

    return final_coords

def generate_new_slice(data,optic_disc_coords,fovea_coords):
    """"""
    from napari_cool_tools_vol_proc._projection_tools_funcs import projection, ProjectionType, ProjectionDir

    en_face_data = projection(data,projection_type=ProjectionType.MEAN.value,axis=ProjectionDir.EN_FACE.value)

    en_face_slice = get_full_line_coordinates(optic_disc_coords,fovea_coords,en_face_data.shape[0],en_face_data.shape[1])
