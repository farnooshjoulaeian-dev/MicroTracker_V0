import cv2
import pandas as pd

from skimage import filters
from skimage import morphology
from skimage import measure


def detect_cells_in_frame(
    frame,
    frame_number,
    pixel_size=1.178,
    background_sigma=50,
    closing_radius_px=2,
    min_area_px=6,
    max_area_px=120
):
    # Convert color video frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Estimate smooth illumination background
    background = filters.gaussian(
        gray_frame,
        sigma=background_sigma,
        mode="nearest",
        preserve_range=True
    )

    # Dark cells become bright after subtraction
    corrected = background - gray_frame

    # Segment cells using Otsu threshold
    threshold_value = filters.threshold_otsu(corrected)
    thresholded_img = corrected > threshold_value

    # Merge small separated parts of the same cell
    closed_mask = morphology.binary_closing(
        thresholded_img,
        morphology.disk(closing_radius_px)
    )

    # Label connected objects
    label_image = measure.label(
        closed_mask,
        connectivity=closed_mask.ndim
    )

    rows = []

    # Measure objects and keep only plausible cell sizes
    for region in measure.regionprops(label_image, intensity_image=corrected):
        area_px = region.area

        if area_px < min_area_px or area_px > max_area_px:
            continue

        rows.append({
            "frame": frame_number,
            "label": region.label,

            "y_px": region.centroid[0],
            "x_px": region.centroid[1],
            "y_um": region.centroid[0] * pixel_size,
            "x_um": region.centroid[1] * pixel_size,

            "area_px": area_px,
            "area_um2": area_px * pixel_size**2,
            "mean_intensity": region.intensity_mean,
            "threshold_value": threshold_value,

            "eccentricity_value": region.eccentricity,
            "orientation_value": region.orientation,
            "axis_major_length": region.axis_major_length,
            "axis_minor_length": region.axis_minor_length,
        })
        
        
        

    features = pd.DataFrame(rows)

    return features, corrected, thresholded_img, closed_mask, label_image



