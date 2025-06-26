import numpy as np
import polarTransform
from matplotlib import pyplot as plt
from napari.types import ImageData
from napari_cool_tools_img_proc._equalization_funcs import normalize_data_in_range_func
from skimage import io
from skimage.transform import rotate

# import cv2


def cartify(image_path):
    image = io.imread(image_path)
    # print(image.shape)
    width, height, channel = image.shape

    image = rotate(
        image, 90, resize=True
    )  # to turn theta axis to Y axis as needed for polarTransform
    height_pad = int(
        round(1.66 * height * 2)
    )  # estimates for being in water and typical ref arm position
    image = np.pad(
        image, [(0, 0), (height_pad, 0), (0, 0)], mode="constant", constant_values=0
    )

    # print(image.shape)

    cart_image, ptSettings = polarTransform.convertToCartesianImage(
        image,
        initialAngle=-51 * np.pi / 180,
        finalAngle=51 * np.pi / 180,
        hasColor=True,
    )

    cart_image = rotate(cart_image, -90, resize=True)
    # print(cart_image.shape)
    cart_image = cart_image[:, :, :3]
    # print(cart_image.shape)
    cart_image = np.clip(cart_image, 0, 1)
    plt.imshow(cart_image)
    plt.show()
    plt.imsave("cartesian.png", cart_image)

    return cart_image, ptSettings


def cartify_numpy(img_data: ImageData) -> ImageData:
    # image = io.imread(image_path)
    # print(image.shape)

    print(f"img_data shape: {img_data.shape}")
    image = normalize_data_in_range_func(img_data, min_val=0.0, max_val=255.0).astype(
        np.uint8
    )
    image = np.tile(image, (3, 1, 1, 1)).transpose(1, 2, 3, 0)
    print(f"image shape: {image.shape}")

    width, height, channel = image.shape

    image = rotate(
        image, 90, resize=True
    )  # to turn theta axis to Y axis as needed for polarTransform
    height_pad = int(
        round(1.66 * height * 2)
    )  # estimates for being in water and typical ref arm position
    image = np.pad(
        image, [(0, 0), (height_pad, 0), (0, 0)], mode="constant", constant_values=0
    )

    # print(image.shape)

    cart_image, ptSettings = polarTransform.convertToCartesianImage(
        image,
        initialAngle=-51 * np.pi / 180,
        finalAngle=51 * np.pi / 180,
        hasColor=True,
    )

    cart_image = rotate(cart_image, -90, resize=True)
    # print(cart_image.shape)
    cart_image = cart_image[:, :, :3]
    # print(cart_image.shape)
    cart_image = np.clip(cart_image, 0, 1)
    plt.imshow(cart_image)
    plt.show()
    plt.imsave("cartesian.png", cart_image)

    return cart_image, ptSettings


# Just replace image-path below with the full path for your image path
# Saves the full res image to the folder this code is saved in as "cartesian.png"
image_path = r"C:\Users\ThreadRipper_01\Downloads\MW_CME_.png"

cartify(image_path)
