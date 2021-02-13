#==============================================================================#
#  Author:       Dominik Müller                                                #
#  Copyright:    2021 IT-Infrastructure for Translational Medical Research,    #
#                University of Augsburg                                        #
#                                                                              #
#  This program is free software: you can redistribute it and/or modify        #
#  it under the terms of the GNU General Public License as published by        #
#  the Free Software Foundation, either version 3 of the License, or           #
#  (at your option) any later version.                                         #
#                                                                              #
#  This program is distributed in the hope that it will be useful,             #
#  but WITHOUT ANY WARRANTY; without even the implied warranty of              #
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the               #
#  GNU General Public License for more details.                                #
#                                                                              #
#  You should have received a copy of the GNU General Public License           #
#  along with this program.  If not, see <http://www.gnu.org/licenses/>.       #
#==============================================================================#
#-----------------------------------------------------#
#                   Library imports                   #
#-----------------------------------------------------#
# External libraries
from albumentations import Compose
from albumentations import CenterCrop, Crop
# Internal libraries/scripts
from aucmedi.data_processing.subfunctions.sf_base import Subfunction_Base

#-----------------------------------------------------#
#           Subfunction class: Retinal Crop           #
#-----------------------------------------------------#
""" Retinal cropping function for the specific microscopes utilized in the RIADD challenge.

Methods:
    __init__                Object creation function
    transform:              Crop retinal image.
"""
class Retinal_Crop(Subfunction_Base):
    #---------------------------------------------#
    #                Initialization               #
    #---------------------------------------------#
    def __init__(self):
        # Initialize center cropping transform for each microscope
        self.cropper_alpha = Compose([CenterCrop(width=1424,
                                                 height=1424,
                                                 p=1.0, always_apply=True)])
        self.cropper_beta = Compose([CenterCrop(width=1536,
                                                height=1536,
                                                p=1.0, always_apply=True)])
        self.cropper_gamma = Compose([Crop(x_min=248, x_max=3712,
                                           y_min=408, y_max=3872,
                                           p=1.0, always_apply=True)])

    #---------------------------------------------#
    #                Transformation               #
    #---------------------------------------------#
    def transform(self, image):
        # Microscope: TOPCON 3D OCT-2000
        if image.shape == (2144, 2144, 3):
            image_cropped = self.cropper_alpha(image=image)["image"]
        # TOPCON TRC-NW300
        elif image.shape == (2048, 2048, 3):
            image_cropped = self.cropper_beta(image=image)["image"]
        # Microscope: Kowa VX – 10α
        elif image.shape == (4288, 4288, 3):
            image_cropped = self.cropper_gamma(image=image)["image"]
        # Return resized image
        return image_cropped
