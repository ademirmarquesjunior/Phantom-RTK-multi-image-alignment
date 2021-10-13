# -*- coding: utf-8 -*-
"""
Created on Wed Oct  6 00:00:25 2021

@author: adeju
"""

import cv2
import numpy as np
import rasterio as rt
from rasterio.enums import Resampling
import matplotlib.pyplot as plt
import glob
import os
from shutil import copyfile


import PIL.Image as Image



def load_image(file, scale):
    dataset = rt.open(file)

    data = dataset.read(
        out_shape=(
            dataset.count,
            int(dataset.height * scale),
            int(dataset.width * scale)
        ),
        resampling=Resampling.bilinear
    )

    transform = dataset.transform * dataset.transform.scale(
        (dataset.width / data.shape[-1]),
        (dataset.height / data.shape[-2])
    )

    image = np.moveaxis(data, 0, -1)
    crs = dataset.profile['crs']
    width = np.shape(image)[1]
    height = np.shape(image)[0]
    count = np.shape(image)[2]

    new_dataset = rt.open("temp.tif", 'w', driver='GTiff',
                          height=height, width=width,
                          count=count, dtype=str(image.dtype),
                          crs=crs,
                          transform=transform)

    return image, new_dataset


def save_image(image, file, crs, transform):
    width = np.shape(image)[1]
    height = np.shape(image)[0]

    try:
        count = np.shape(image)[2]
        array = np.moveaxis(image, 2, 0)
    except Exception:
        count = 1
        array = np.reshape(image, (1, np.shape(image)[0],
                                   np.shape(image)[1]))

    new_dataset = rt.open(file, 'w', driver='GTiff',
                          height=height, width=width,
                          count=count, dtype=str(array.dtype),
                          crs=crs,
                          transform=transform)

    new_dataset.write(array)
    new_dataset.close()

    return


def rescale(data, range=(0,1)):
  return np.interp(data, (data.min(), data.max()), range)


# Read the images to be aligned
base_image, dataset = load_image("F:/Unisinos/Parreira/Phantom_10m/Phantom_10m/DJI_0011.TIF", 1)
target_image, dataset = load_image("F:/Unisinos/Parreira/Phantom_10m/Phantom_10m/DJI_0012.TIF", 1)

# Convert images to grayscale
# base_image_gray = cv2.cvtColor(base_image, cv2.COLOR_BGR2GRAY)
# target_image_gray = cv2.cvtColor(target_image, cv2.COLOR_BGR2GRAY)


def ECC_alignment(base_image, target_image):
    
    # https://learnopencv.com/image-alignment-ecc-in-opencv-c-python/

    base_image = np.resize(base_image, (np.shape(base_image)[0], np.shape(base_image)[1]))
    target_image = np.resize(target_image, (np.shape(target_image)[0], np.shape(target_image)[1]))

    # base_image = rescale(base_image, range=(0, 1))
    # target_image = rescale(target_image, range=(0, 1))

    base_image = base_image/100000
    target_image = target_image/100000

    base_image = np.asarray(base_image, dtype=np.float32) # np.float32
    target_image = np.asarray(target_image, dtype=np.float32)

    # Find size of image1
    sz = base_image.shape

    # Define the motion model
    warp_mode = cv2.MOTION_TRANSLATION

    # Define 2x3 or 3x3 matrices and initialize the matrix to identity
    if warp_mode == cv2.MOTION_HOMOGRAPHY:
        warp_matrix = np.eye(3, 3, dtype=np.float32)
    else:
        warp_matrix = np.eye(2, 3, dtype=np.float32)

    # Specify the number of iterations.
    number_of_iterations = 5000

    # Specify the threshold of the increment
    # in the correlation coefficient between two iterations
    termination_eps = 1e-10

    # Define termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                number_of_iterations,  termination_eps)

    # Run the ECC algorithm. The results are stored in warp_matrix.
    (cc, warp_matrix) = cv2.findTransformECC(base_image, target_image, warp_matrix,
                                             warp_mode, criteria, None, 1)

    if warp_mode == cv2.MOTION_HOMOGRAPHY:
        # Use warpPerspective for Homography
        target_image_aligned = cv2.warpPerspective(target_image, warp_matrix,
                                          (sz[1],sz[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
    else:
        # Use warpAffine for Translation, Euclidean and Affine
        target_image_aligned = cv2.warpAffine(target_image, warp_matrix, (sz[1],sz[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP);

    target_image_aligned = np.uint16(target_image_aligned*100000)

    return target_image_aligned




base_band = 1

# output_path = 'F:/Unisinos/Parreira/Phantom_10m/Phantom_10m/aligned/'

# input_path = 'F:/Unisinos/Parreira/Phantom_10m/Phantom_10m/'



# input_path = 'F:/Unisinos/Parreira/Phantom_5m/Phantom_5m/102FPLAN/'

# output_path = 'F:/Unisinos/Parreira/Phantom_5m/Phantom_5m/102FPLAN/aligned/'


input_path = 'F:/Unisinos/Parreira/Phantom_40m/40m/104FPLAN/'

output_path = 'F:/Unisinos/Parreira/Phantom_40m/40m/104FPLAN/aligned/'


os.chdir(input_path)

base_images = glob.glob('DJI_***' + str(base_band) + '.TIF')

target_images = []

for ref in base_images:
    base_file = list(ref)
    base_file[7] = '*'
    base_file = "".join(base_file)

    aux = glob.glob(base_file)

    aux.pop(base_band-1)

    # target_images.append(aux)
    
    # Open file with Rasterio
    # base_image, dataset_input = load_image(ref, 1)
    
    # Open file with Pillow
    base_image = Image.open(ref)
    ref_tag = base_image.tag_v2
    
    
    # base_image.save(output_path+ref, tiffinfo=ref_tag)
    
    # Copy and save the file instead of open and save
    #copyfile(input_path+ref, output_path+ref)
    
    
    #save_image(base_image, output_path+ref, dataset_input.crs, dataset_input.transform)

    for target in aux:
        #target_image, dataset_target = load_image(target, 1)
        
        target_image = Image.open(target)
        tag_v2 = target_image.tag_v2
        
        target_image = np.asarray(target_image)
        
        try:
            aligned_image = ECC_alignment(base_image, target_image)
        except Exception as e:
            print(e)
            print("Image " + target + " did not align.")
            aligned_image = target_image
        
        aligned_image = Image.fromarray(aligned_image)
        aligned_image.tag_v2 = tag_v2
        
        aligned_image.save(output_path+target, tiffinfo=tag_v2)
        
        
        #save_image(aligned_image, output_path+target, dataset_target.crs, dataset_target.transform)
