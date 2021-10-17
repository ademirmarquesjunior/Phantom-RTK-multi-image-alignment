# -*- coding: utf-8 -*-
"""
Created on Thu Jul 15 15:47:11 2021

@author: adeju
"""

# from PyQt5 import sip
from PyQt5 import uic, QtWidgets
# from PyQt5.QtGui import QPixmap, QImage
import sys
import cv2
import numpy as np
from PIL import Image

# import rasterio as rt
# from rasterio.enums import Resampling
# import matplotlib.pyplot as plt
import glob
import os
from shutil import copy


# def load_image(file, scale):
#     dataset = rt.open(file)

#     data = dataset.read(
#         out_shape=(
#             dataset.count,
#             int(dataset.height * scale),
#             int(dataset.width * scale)
#         ),
#         resampling=Resampling.bilinear
#     )

#     transform = dataset.transform * dataset.transform.scale(
#         (dataset.width / data.shape[-1]),
#         (dataset.height / data.shape[-2])
#     )

#     image = np.moveaxis(data, 0, -1)
#     crs = dataset.profile['crs']
#     width = np.shape(image)[1]
#     height = np.shape(image)[0]
#     count = np.shape(image)[2]

#     new_dataset = rt.open("temp.tif", 'w', driver='GTiff',
#                           height=height, width=width,
#                           count=count, dtype=str(image.dtype),
#                           crs=crs,
#                           transform=transform)

#     return image, new_dataset


# def save_image(image, file, crs, transform):
#     width = np.shape(image)[1]
#     height = np.shape(image)[0]

#     try:
#         count = np.shape(image)[2]
#         array = np.moveaxis(image, 2, 0)
#     except Exception:
#         count = 1
#         array = np.reshape(image, (1, np.shape(image)[0],
#                                    np.shape(image)[1]))

#     new_dataset = rt.open(file, 'w', driver='GTiff',
#                           height=height, width=width,
#                           count=count, dtype=str(array.dtype),
#                           crs=crs,
#                           transform=transform)

#     new_dataset.write(array)
#     new_dataset.close()

#     return


# def rescale(data, range=(0,1)):
#   return np.interp(data, (data.min(), data.max()), range)


def ECC_alignment(base_image, target_image):
    # https://learnopencv.com/image-alignment-ecc-in-opencv-c-python/

    base_image = np.resize(base_image, (np.shape(base_image)[0],
                                        np.shape(base_image)[1]))
    target_image = np.resize(target_image, (np.shape(target_image)[0],
                                            np.shape(target_image)[1]))

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


class Ui(QtWidgets.QMainWindow):
    def __init__(self):
        super(Ui, self).__init__()
        uic.loadUi('view.ui', self)
        self.input_path = ''
        self.output_path = ''
        self.init_Ui()
        self.show()

    def init_Ui(self):
        # self.actionOpen_Image.triggered.connect(self.open_folder)
        # self.actionOpen_Superpixel_file.triggered.connect(self.open_superpixel)

        self.openInputFolder.clicked.connect(lambda: self.open_folder(0))
        self.selectOutputFolder.clicked.connect(lambda: self.open_folder(1))
        self.alignButton.clicked.connect(self.align_photos)
        self.progressBar.setValue(0)
        self.statusBar().showMessage("Idle")

    def open_folder(self, mode):
        path = QtWidgets.QFileDialog.getExistingDirectory(self, "Open a folder")
        if path != ('', ''):
            if mode == 0:
                self.input_path = path + '/'
                self.labelInputFolder.setText(self.input_path)
            if mode == 1:
                self.output_path = path + '/'
                self.labelOutputFolder.setText(self.output_path)

    def align_photos(self):
        base_band = 1

        os.chdir(self.input_path)

        base_images = glob.glob('DJI_***' + str(base_band) + '.TIF')

        progress_bar_max = np.size(base_images)
        progress_bar_count = 0
        self.progressBar.setValue(0)
        self.statusBar().showMessage("Processing")

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
            base_image = Image.open(self.input_path+ref)
            ref_tag = base_image.tag_v2

            try:
                # base_image.save(self.output_path+ref, tiffinfo=ref_tag)
                # Copy and save the file instead of open and save
                copy(self.input_path+ref, self.output_path+ref)
            except Exception as e:
                print(e)

            # save_image(base_image, output_path+ref, dataset_input.crs, dataset_input.transform)

            for target in aux:

                # target_image, dataset_target = load_image(target, 1)

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

                aligned_image.save(self.output_path+target, tiffinfo=tag_v2)
            
            progress_bar_count += 1
            
            self.progressBar.setValue(int((progress_bar_count/progress_bar_max)*100))
        self.statusBar().showMessage("Done!")


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    window = Ui()
    app.quit()
