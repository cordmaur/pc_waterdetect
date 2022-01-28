import matplotlib.pyplot as plt
from pathlib import Path
from lxml import etree
import numpy as np
import planetary_computer as pc
import requests


class DWGlintProcessor:

    adjustable_bands = ['B12']
    
    def __init__(self, img_item, limit_angle=30):
        """
        Create a GLINT processor for the img_item (STAC Item image)
        img_dataset is the XArray dataset with all the bands already loaded (it's optional)
        In intialization, it will retrieve the metadata and create the glint array (22x22 array)
        """
        self.metadata = DWGlintProcessor.get_metadata(img_item, 'granule-metadata')
        
        self.img_item = img_item
        self.limit_angle = limit_angle

        try:
            self.glint_array = self.create_glint_array(self.metadata)

        except BaseException as err:
            self.glint_array = None
            print(f'### GLINT PROCESSOR ERROR #####')
            print(err)

    @staticmethod
    def get_metadata(img_item, metadata):
        print(f'Getting metadata: {metadata}')
        signed_metadata = pc.sign(img_item.assets[metadata].href)
        results = requests.get(signed_metadata)
        return results.content.decode('utf-8')


    @staticmethod
    def get_grid_values_from_xml(tree_node, xpath_str):
        """Receives a XML tree node and a XPath parsing string and search for children matching the string.
           Then, extract the VALUES in <values> v1 v2 v3 </values> <values> v4 v5 v6 </values> format as numpy array
           Loop through the arrays to compute the mean.
        """
        node_list = tree_node.xpath(xpath_str)

        arrays_lst = []
        for node in node_list:
            values_lst = node.xpath('.//VALUES/text()')
            values_arr = np.array(list(map(lambda x: x.split(' '), values_lst))).astype('float')
            arrays_lst.append(values_arr)

        return np.nanmean(arrays_lst, axis=0)

    @staticmethod
    def create_glint_array(metadata):

        # Get the XML root element (encode it in ascii beforehand)
        root = etree.XML(metadata.encode('ascii'))

        sun_angles = 'Sun_Angles_Grid'

        sun_zenith = np.deg2rad(DWGlintProcessor.get_grid_values_from_xml(root, f'.//{sun_angles}/Zenith'))[:-1, :-1]
        sun_azimuth = np.deg2rad(DWGlintProcessor.get_grid_values_from_xml(root, f'.//{sun_angles}/Azimuth'))[:-1,:-1]

        view_zenith = np.deg2rad(
            DWGlintProcessor.get_grid_values_from_xml(root, './/Viewing_Incidence_Angles_Grids/Zenith'))[:-1, :-1]
        view_azimuth = np.deg2rad(
            DWGlintProcessor.get_grid_values_from_xml(root, './/Viewing_Incidence_Angles_Grids/Azimuth'))[:-1, :-1]

        phi = sun_azimuth - view_azimuth
        Tetag = np.cos(view_zenith) * np.cos(sun_zenith) - np.sin(view_zenith) * np.sin(sun_zenith) * np.cos(phi)

        # convert results to degrees
        glint_array = np.degrees(np.arccos(Tetag))
        return glint_array

    @staticmethod
    def create_annotated_heatmap(hm, img=None, cmap='magma_r', vmin=0.7, vmax=0.9, ax=None):
        '''Create an annotated heatmap. Parameter img is an optional background img to be blended'''
        
        if ax is None:
            _, ax = plt.subplots(figsize=(15, 15))

        if img is not None:
            ax.imshow(img, alpha=1., extent=(-0.5, 21.5, 21.5, -0.5))

        ax.imshow(hm, alpha=0.5, vmin=vmin, vmax=vmax, cmap=cmap)


        # Loop over data dimensions and create text annotations.
        for i in range(0, hm.shape[0]):
            for j in range(0, hm.shape[1]):
                text = ax.text(j, i, round(hm[i, j], 1),
                               ha="center", va="center", color="cornflowerblue", size=9)

    @staticmethod
    def nn_interpolate(arr, new_size):
        """
        Vectorized Nearest Neighbor Interpolation
        From post: https://gist.github.com/KeremTurgutlu/68feb119c9dd148285be2e247267a203
        """

        old_size = arr.shape
        row_ratio, col_ratio = np.array(new_size) / np.array(old_size)

        # row wise interpolation
        row_idx = (np.ceil(range(1, 1 + int(old_size[0] * row_ratio)) / row_ratio) - 1).astype(int)

        # column wise interpolation
        col_idx = (np.ceil(range(1, 1 + int(old_size[1] * col_ratio)) / col_ratio) - 1).astype(int)

        final_matrix = arr[:, row_idx][col_idx, :]

        return final_matrix


    def create_glint_heatmap(self, img_dataset, brightness=5., ax=None):
        """Create a heatmap with the RGB obtained from the image dataset"""
        
        # create a RGB cube
        rgb = img_dataset[['B04', 'B03', 'B02']].to_array().transpose('y', 'x', 'variable')[::10, ::10] * brightness
        
        # create glint angle
        no_glint_mask = (self.glint_array >= self.limit_angle) | np.isnan(self.glint_array)
        glint_angles = np.where(no_glint_mask, np.nan, self.glint_array)
        
        # get the minimum angle
        min_angle = np.nanmin(self.glint_array)
        
        # create the heatmap with the RGB as background
        DWGlintProcessor.create_annotated_heatmap(hm=glint_angles,
                                                  img=rgb, 
                                                  vmin=min_angle if min_angle < self.limit_angle else self.limit_angle,
                                                  vmax=self.limit_angle,
                                                  ax=ax)
        
    def create_multiplication_coefs(self, min_multiplier=1.2, max_multiplier=2.2):
        # create a matrix with the multiplication coeficients to be applied directly to the threshold value

        # max angle variation to consider the multiplier
        angle_range = 0 - self.limit_angle
        # multiplier range
        multiplier_range = max_multiplier - min_multiplier
        
        multiplier = np.where(self.glint_array > self.limit_angle, 
                              0, 
                              (self.glint_array - self.limit_angle) / angle_range * multiplier_range + min_multiplier)
        return multiplier

    def show_multiplication_coefs(self):
        return DWGlintProcessor.create_annotated_heatmap(self.create_multiplication_coefs(), vmin=1, vmax=3)

    def glint_adjusted_threshold(self, value, mask=None, min_multiplier=1.2, max_multiplier=2.2, 
                                 out_shape=(10980, 10980)):
        """
        Create a threshold array, given a base threshold value and minimum and maximum multipliers
        value: Basic threshold value. This will be the value for Angle > limit angle (no possible glint)
        min_multiplier: minimum multiplier to be applied when the angle enters the 30deg limit
        max_multiplier: maximum multipliert to be applied when the angle is 0 (100% chance of glint)
        """

        # check if it is possible to ajust the threshold. If it is not, return the plain value
        # check the following conditions:
        # 1- if the glint_arry do exists (it can have an error)
        # 2- if there is any possible glint in the scene
        if (self.glint_array is None) or (np.nanmin(self.glint_array) > self.limit_angle):
            return value

        # create a grid with the multiplication coeficients
        thresh_grid = value * self.create_multiplication_coefs(min_multiplier=min_multiplier, max_multiplier=max_multiplier)

        thresh_array = DWGlintProcessor.nn_interpolate(thresh_grid, out_shape)
        return thresh_array[~mask] if mask is not None else thresh_array

    def __repr__(self):
        s = f'Glint Processor'
        return s
