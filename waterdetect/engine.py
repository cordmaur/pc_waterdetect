import planetary_computer as pc
import rasterio as rio
import rioxarray as xrio
import xarray as xr
import numpy as np
import pandas as pd
from pathlib import Path
from time import sleep

from sklearn import cluster
from sklearn.metrics import calinski_harabasz_score
from sklearn.naive_bayes import GaussianNB
from skimage.morphology import binary_opening, diamond

from joblib import Parallel, delayed
from functools import partial
import matplotlib.pyplot as plt

# PACKAGE IMPORTS
from .cloudless import get_gee_img, create_cloud_mask
from .glint import DWGlintProcessor
from geeS2downloader import GEES2Downloader

from concurrent.futures import ThreadPoolExecutor



def reshape(array, out_shape):
    return array.rio.reproject(array.rio.crs, shape=out_shape, resampling=rio.enums.Resampling.nearest)


def open_image(image, bands, out_shape=(10980, 10980), n_jobs=2, retries=3):
    '''Open bands of an image and output them as a cube XArray with the desired out_shape'''
    
    print(f'Getting image: {image.id}')
    # sign the assets from the image
    while retries > 0:
        try:
            # sign the assets to grant access to them
            assets = {band: pc.sign(image.assets[band].href) for band in bands}

            # open the datasets (bands)
            datasets = {band: xrio.open_rasterio(asset).squeeze() for band, asset in assets.items()}

            # downaload and rescale the bands correctly
            # this part may have connection errors and it will retry if not succeed
            for band in datasets:
                print(f'Downloading {band}')
                if band != 'SCL':
                    datasets[band] = (datasets[band]/10000).astype('float32')
                else:
                    datasets[band] = (datasets[band]*1).astype('uint8')

            retries = 0
        
        # if any problem occurred, will repeat the whole process
        except Exception as e:
            print(f'Problem downloading band {band}.')
            print(e)
            
            # decrease the retries counter
            retries -= 1

            if retries > 0:
                print(f'Waiting 60sec. {retries} Retries left')
                sleep(61)
            else:
                print('Skipping...')
                raise Exception(f'Band {band} could not be downloaded correctly')
                    
    # get the arrays to reshape
    reshape_arrays = {band: array for band, array in datasets.items() if array.shape != out_shape}
    
    print('Finished downloading. Will reshape in parallel...')
    
    reshaped_arrays = Parallel(n_jobs, backend='threading')(delayed(reshape)(reshape_array, out_shape) for reshape_array in reshape_arrays.values())
    for band, reshaped in zip(reshape_arrays.keys(), reshaped_arrays):
        print(f'resampling {band} to {out_shape}')
        datasets[band] = reshaped
    
    # with Pool(n_jobs) as p:
    #     reshaped_arrays = p.map(partial(reshape, out_shape=out_shape), reshape_arrays.values())
    #     for band, reshaped in zip(reshape_arrays.keys(), reshaped_arrays):
    #         print(f'resampling {band} to {out_shape}')
    #         datasets[band] = reshaped

    return xr.Dataset(datasets)


def bands_to_columns(bands, mask):
    np_mask = mask.values

    # create a list with the bands as columns
    list_columns = [bands[band].values[~mask].reshape(-1, 1) for band in bands]
    

    # combine them into a 2D matrix (columns representing each band and only valid pixels)
    np_columns = np.concatenate(list_columns, axis=1)
    
    # multiply bands by 4
    for column, band in enumerate(bands):
        if band not in ['mndwi', 'ndwi', 'mbwi', 'SCL']:
            np_columns[:, column] = np_columns[:, column] * 4
        

    # return as an xarray
    return xr.DataArray(np_columns, dims=['data', 'band'], coords={'band': list(bands.keys())})


def clusterize(data, k, ret='metric'):
    '''Apply cluster and return the labels or the metric or both
    ret = metric or labels or both'''
    cluster_model = cluster.AgglomerativeClustering(n_clusters=k, linkage='average')

    labels = cluster_model.fit_predict(data).astype('int8')
    metric = calinski_harabasz_score(data, labels)

    if ret == 'metric':
        return metric
    elif ret == 'labels':
        return labels
    else:
        return (labels, metric)
        
        
def find_best_k(data, cluster_bands, min_k=2, max_k=7, n_jobs=1):
    train_data = data.sel(band=cluster_bands)
    
    metrics = Parallel(n_jobs=n_jobs)(delayed(clusterize)(train_data, k) for k in range(min_k, max_k+1))

    for k, metric in enumerate(metrics):
        print(f'k={k+min_k} - Calinski={metric}')
        
    return metrics.index(max(metrics)) + min_k


def calc_clusters_params(data, labels):

    df = pd.DataFrame()
    
    for label in np.unique(labels):
        
        indexer = (labels == label)
        cluster = data[indexer, :]

        mean = pd.Series(cluster.mean(dim='data'), index=cluster.band, name=('mean', label))
        std = pd.Series(cluster.std(dim='data'), index=cluster.band, name=('std', label))
        count = pd.Series(cluster.count(dim='data'), index=cluster.band, name=('count', label))
        
        df = pd.concat([df, mean, std, count], axis=1)

    df = df.T
    df.index = pd.MultiIndex.from_tuples(df.index)
    return df


def itendify_water_cluster(params, band='B08', rule='min'):
    
    if rule == 'min':
        return params.loc['mean', band].argmin()
    if rule == 'max':
        return params.loc['mean', band].argmax()
    

def open_s2_bands(image, bands, ref_band='B03'):
    '''A faster version then open_datasets that uses a reference band as output shape and coordinates to speedup process
    with this version, the output shape may not be completely arbitrary and must match an existing S2 band'''
    
    # open the ref_band
    ref_array = xrio.open_rasterio(pc.sign(image.assets[ref_band].href))
    
    # sign the assets from the image
    assets = {band: pc.sign(image.assets[band].href) for band in bands}
   
    # open the datasets (bands) as arrays (in the correct shape)
    arrays = {band: rio.open(asset).read(out_shape=ref_array.shape) for band, asset in assets.items()}
    
    # create the xarrays
    xarrays = {band: xr.DataArray(array, 
                                  dims=['band', 'y', 'x'], 
                                  coords={'band': [band], 'x': ref_array.coords['x'], 'y': ref_array.coords['y']}) 
               for band, array in arrays.items()}
    
    return xarrays


def calc_normalized_difference(img1, img2, mask=None, compress_cte=0.02):
    """
    Calc the normalized difference of given arrays (img1 - img2)/(img1 + img2).
    Updates the mask if any invalid numbers (ex. np.inf or np.nan) are encountered
    :param img1: first array
    :param img2: second array
    :param mask: initial mask, that will be updated
    :param compress_cte: amount of index compression. The greater, the more the index will be compressed towards 0
    :return: nd array filled with -9999 in the mask and the mask itself
    """

    # changement for negative SRE scenes
    # ##### UPDATED ON 01/04/2021

    # create a minimum array
    # min_values = np.where(img1 < img2, img1, img2)

    # then create the to_add matrix (min values turned into positive + epsilon)
    # min_values = np.where(min_values <= 0, -min_values + 0.001, 0) + compress_cte
    img1 = img1.astype('float16') + compress_cte
    img2 = img2.astype('float16') + compress_cte
    nd = ((img1) - (img2)) / ((img1) + (img2))
    
    return nd
    

def generalize(train_data, train_labels, all_data, train_bands):
    model = GaussianNB()
    
    model.fit(train_data.sel(band=train_bands), train_labels)
    
    labels = model.predict(all_data.sel(band=train_bands))
    
    return labels


# ********************************************************************************************
class WaterDetect:
    load_bands = ['B02', 'B03', 'B04', 'B11', 'B12', 'B08', 'SCL']
        
    colormaps = {'SCL': 'tab20',
                 'watermask': 'winter_r',
                 'mask': 'RdGy_r',
                 'clusters': 'Set1'}
    
    def __init__(self, img_item, cluster_bands, s2clouds=False, sampling=10000, out_shape=(10980, 10980), max_k=7, n_jobs=4):
        
        self.img_item = img_item

        with ThreadPoolExecutor(max_workers=2) as executor:
            # Just launch s2clouds if demanded
            load_clouds = executor.submit(self.get_s2cloudmask) if s2clouds else None

            # open image and GLINT processor will be launched anyways
            load_dataset = executor.submit(open_image, image=img_item, out_shape=out_shape, bands=WaterDetect.load_bands, n_jobs=2)
            glint_proc = executor.submit(DWGlintProcessor, img_item=img_item)

        # The pool will exit the context if everything has finished
        self.glint_proc = glint_proc.result()                
        self.img = load_dataset.result()

        # check for the s2clouds from GEE
        clouds_array = load_clouds.result() if s2clouds else None
        
        if clouds_array is not None:
            # georreference the output array
            geoarray = WaterDetect.georeference_array(dataset=self.img, 
                                                        array=clouds_array, 
                                                        name='s2cloudmask', 
                                                        dtype='bool')

            # append it to the images dataset
            self.img = self.img.assign({'s2cloudmask': geoarray})

        # Init variables
        self.cluster_bands = cluster_bands
        self.sampling = sampling
        self.out_shape = out_shape
        self.max_k = max_k
        self.n_jobs = n_jobs

        # Init result variables
        self.water_cluster = None
        self.cluster_params = None        
        
        self.calc_indices()
        
    @staticmethod
    def georeference_array(dataset, array, name, dtype='int'):
        """
        Create an xarray with coordinates and CRS system that corresponds the dataset
        """
        # create the transformation matrix using the bounds of the dataset
        transform = rio.transform.from_bounds(*dataset.rio.bounds(), *array.shape)

        # create the x and y coordinates with this transformation
        coords_x, coords_y = rio.transform.xy(transform, rows=range(array.shape[0]), cols=range(array.shape[1]))

        # create the DataArray using the coordinates
        dtype = 'uint8' if dtype == 'bool' else dtype
        geoarray = xr.DataArray(array.astype(dtype), dims=['y', 'x'], coords={'x': coords_x, 'y': coords_y}, name=name)

        # write crs to the DataArray
        geoarray = geoarray.rio.write_crs(dataset.rio.crs)

        # interpolate to the final resolution
        geoarray = geoarray.rio.reproject(dst_crs=geoarray.rio.crs, resolution=dataset.rio.resolution()[0],
                                          resampling=rio.enums.Resampling.nearest)

        # append to the dataset
        return geoarray.rename(name)
        
    def get_s2cloudmask(self):
        """Use GEE API to download and preprocess the clouds and shadows using S2 Cloud Probability"""
        
        print(f'Downloading s2cloudless for {self.img_item.id}')
        
        # Get the GEE Image and Cloud Probability
        gee_img = get_gee_img(self.img_item, 'COPERNICUS/S2_SR')
        s2cloud = get_gee_img(self.img_item, 'COPERNICUS/S2_CLOUD_PROBABILITY')

        if gee_img is not None and s2cloud is not None:
            # create the cloud/shadow mask on the GEE environment
            mask = create_cloud_mask(gee_img, s2cloud, cloud_thresh=50)

            # Once finished, load it in memory using geeS2Downloader
            downloader = GEES2Downloader()
            downloader.download(mask, band='cloudmask')
            mask_array = downloader.array.astype('uint8')

            return mask_array
        
        else:
            print(f'Problem to find clouds for {self.img_item.id}. Proceeding with SEN2COR clouds')
            return None

    def _calc_indices(self, index_name):
        if index_name == 'mndwi':
            index = calc_normalized_difference(self.img['B03'], self.img['B12'])
        
        elif index_name == 'ndwi':
            index = calc_normalized_difference(self.img['B03'], self.img['B08'])
        
        elif index_name == 'mbwi':
            index = (2*self.img['B03']/10000 - self.img['B04']/10000 - self.img['B08']/10000 - self.img['B11']/10000 - self.img['B12']/10000).astype('float16')
        
        elif index_name == 'mask':
            # if s2cloudmask is present, use it's cloud mask as reference
            if 's2cloudmask' in self.img:
                index = (self.img['SCL'] == 0) | (self.img['s2cloudmask'] == 1)
                
            # otherwise, let's use the SCL layer
            else:
                index = (self.img['SCL'] >= 8) | (self.img['SCL'] == 3) | (self.img['SCL'] == 0)
            
        return index

    def calc_indices(self):
        
        print('Calculating indices')
        
        # Launch the indices calculation in parallel for performance purposes
        indices = Parallel(n_jobs=self.n_jobs)(delayed(self._calc_indices)(index_name) for index_name in ['mndwi', 'ndwi', 'mbwi', 'mask'])
            
        # merge the indices in the img Dataset
        self.img = self.img.assign({'mndwi': indices[0],
                                    'ndwi': indices[1],
                                    'mbwi': indices[2],
                                    'mask': indices[3],
                                   })       
    
    def get_balanced_train_data(self, columns, min_mndwi=0.1, water_percent=0.1):

        # get the subsets
        water_data = columns.where(columns.sel(band='mndwi') >= min_mndwi).dropna(dim='data', how='all')
        other_data = columns.where(columns.sel(band='mndwi') < min_mndwi).dropna(dim='data', how='all')   

        water_train_idx = np.random.randint(len(water_data), size=int(water_percent*self.sampling))
        water_train_data = water_data[water_train_idx]

        other_train_idx = np.random.randint(len(other_data), size=int((1-water_percent)*self.sampling))
        other_train_data = other_data[other_train_idx]

        train_data = xr.concat([water_train_data, other_train_data], dim='data')

        return train_data

    def run_detect_water(self, cluster_bands=None, retries=2):

        # override with the new cluster_bands (the bands must be loaded previously)
        self.cluster_bands = cluster_bands if cluster_bands is not None else self.cluster_bands

        # first, let's convert all data to columns
        columns = bands_to_columns(self.img, self.img['mask'])

        # extract a random sample with defined size
        train_idxs = np.random.randint(len(columns), size=self.sampling)
        train_data = columns[train_idxs, :]

        while retries > 0:
            try:
                train_labels = self._detect_water_cluster(columns, train_data)

                if not self.valid_water_cluster:
                    raise Exception(f'Water cluster B12 centroid above max threshold')

                # if everything goes fine, apply solution and quit
                # self._apply_clustering(columns, train_data, train_labels)
                break

            except Exception as e:
                print(f'**Problem processing {self.img_item.id}')
                print(e)
                retries -= 1

        # After the retries, check again for the validity
        if not self.valid_water_cluster:
            for min_mndwi in range(0, 40, 5):
                print(f'Trying balanced sampling with min_mndwi={min_mndwi/100}')
                train_data = self.get_balanced_train_data(columns, min_mndwi=min_mndwi/100)
                train_labels = self._detect_water_cluster(columns, train_data)

                if self.valid_water_cluster:
                    print(f'Solved scene with previous config.')
                    break

        if self.valid_water_cluster:
            self._apply_clustering(columns, train_data, train_labels)

        else:
            print('It was not possible to process the scene!!!!')

        # try to free some memory
        for temp in [columns, train_idxs, train_data, train_labels]:
            del temp

    def _apply_clustering(self, columns, train_data, train_labels):
        # generalize the labels for the entire the image and write it to the columns
        labels = generalize(train_data, train_labels, columns, train_bands=self.cluster_bands)
        print(f'Generalized for the whole scene')

        # recreate the final matrices
        # First the matrix with the clusters
        cluster_matrix = xr.DataArray(np.zeros(self.out_shape, dtype='int8'), dims=['y', 'x']) - 1
        cluster_matrix.values[~self.img['mask']] = labels
        
        # then, the final water mask
        water_mask = cluster_matrix == self.water_cluster

        # clipping for high SWIR and eroding single pixels
        b12_threshold = self.glint_proc.glint_adjusted_threshold(value=0.1,
                                                                 out_shape=self.out_shape,
                                                                 min_multiplier=1.2,
                                                                 max_multiplier=2.2
                                                                )
        
        water_mask = water_mask.where(self.img['B12'].values < b12_threshold, other=0)
        water_mask.values = binary_opening(water_mask, diamond(1))
        
        # and a watermask with 255 in nodata positions
        nodata_watermask = water_mask.where(cluster_matrix != -1, other=255).astype('uint8')
        
        # append the final solution
        self.img = self.img.assign({'watermask': water_mask, 'clusters': cluster_matrix, 'nodata_watermask': nodata_watermask})

    def _detect_water_cluster(self, columns, train_data):
        
        # then, search the best K
        k = find_best_k(train_data, self.cluster_bands, min_k=2, max_k=self.max_k, n_jobs=self.n_jobs)
        
        # get the training labels and write it to the training data
        print(f'Final clustering with k={k}')
        train_labels = clusterize(train_data.sel(band=self.cluster_bands), k, ret='labels')

        # calc parameters for each trained cluster
        self.cluster_params = calc_clusters_params(train_data, train_labels)
        
        # identify the water_cluster
        self.water_cluster = itendify_water_cluster(self.cluster_params, band='mbwi', rule='max')
        print(f'Water cluster = {self.water_cluster}')
        
        return train_labels

    @property
    def title(self):
        return self.img_item.id[:38]
        
    @property
    def valid_water_cluster(self):
        # Check if the water cluster is valid (B12 centroid bellow 0.2 (max thresh))
        return self.cluster_params.loc[('mean', self.water_cluster), 'B12'] < 0.2

    # ----------------------------------------------------------------------------------
    # ############################### IO FUNCTIONS ###############################
    def save_geotiff(self, band, filename):
        """Save a band on the dataset as a geotiff to the desired filename"""
        xarr = self.img[band]
        xarr.rio.to_raster(filename, compress='PACKBITS')
        
    # ----------------------------------------------------------------------------------
    # ############################### PLOTTING FUNCTIONS ###############################
    def get_thumbnail(self, band, scale=1):
        """
        Return thumbnail arrays to be displayed with imshow.
        """
        if band == 'rgb':
            thumb = self.img[['B04', 'B03', 'B02']].isel(x=slice(None, None, 10), y=slice(None, None, 10)).to_array(dim='band').transpose('y', 'x', 'band')
            thumb = (thumb * 5).values
            
        else:
            thumb = np.ma.array(data=self.img[band][::10, ::10],
                                mask=self.img['mask'][::10, ::10])

        # eliminate negative values
        if thumb.min() < 0:
            thumb = thumb - thumb.min()

        # check if the array to be ploted is continuous to clip on 1
        if not np.issubdtype(thumb.dtype, np.integer):
            thumb[thumb > 1] = 1
            thumb = thumb.astype('float32')

        # force scale == 1 for SCL band
        scale = 1 if band == 'SCL' else scale
        
        return thumb * scale
    
    def plot_thumb(self, band, ax):
        '''Plot a thumbnail into a specific Matplotlib axis'''

        # if the thumbnail is the glint heatmap, it will be plotted in by the glint processor
        if band.lower() == 'glint':
            self.glint_proc.create_glint_heatmap(img_dataset=self.img, ax=ax)
        
        # otherwise, plot it here
        else:
            # get the thumbnail for the band
            array = self.get_thumbnail(band)

            # Adjust the colormap
            if band in WaterDetect.colormaps:
                cmap = WaterDetect.colormaps[band]
            else:
                cmap = 'viridis'


            # display the image
            ax.imshow(array, cmap=cmap)
            
        ax.set_title(band.upper())

    def plot_thumbs(self, cols=2, thumbs=['rgb', 'mask', 'mndwi', 'ndwi'], save_folder=None):
        plt.style.use('default')
        # calc number of rows
        rows = len(thumbs)//cols + (0 if len(thumbs) % cols == 0 else 1)

        base_width = 15
        base_height = base_width / cols

        fig, ax = plt.subplots(
            nrows=rows, 
            ncols=cols, 
            figsize=(base_width, base_height*rows), 
            num=1, 
            clear=True
        )

        fig.suptitle = self.img_item.id

        for i, thumb in enumerate(thumbs):
            self.plot_thumb(thumb, ax.reshape(-1)[i])
        
        # if a save folder is given, save it there
        if save_folder is not None:
            fig.savefig(Path(save_folder)/f'{self.img_item.id[:38]}_thumbs.png', dpi=300)
            
            # release all the memory
            # fig.clear()
            # plt.close('all')
            # del ax
            # del fig
            
    # ----------------------------------------------------------------------------------
    # ############################### GRAPH FUNCTIONS ###############################
    def plot_clustered_data(self, data, cluster_names, file_name, graph_options, water_cluster, ax=None):
        plt.style.use('seaborn-whitegrid')

        plot_colors = ['goldenrod', 'darkorange', 'tomato', 'brown', 'gray', 'salmon', 'black', 'orchid', 'firebrick','orange', 'cyan']
        # plot_colors = list(colors.cnames.keys())

        if ax is None:
            _, ax = plt.subplots(figsize=(15, 10), dpi=100)

        k = np.unique(data[:, 2]).astype('int8')

        for i in k:
            cluster_i = data[data[:, 2] == i, 0:2]

            if i == water_cluster:
                label = 'Water'
                colorname = 'deepskyblue'
            else:
                label = cluster_names[int(i)]
                colorname = plot_colors[int(i)]

            ax.set_xlabel(graph_options['x_label'])
            ax.set_ylabel(graph_options['y_label'])
            # ax1.set_title(graph_options['title'])

            ax.plot(cluster_i[:, 0], cluster_i[:, 1], '.', label=label, c=colorname,)

        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles, labels)

    def get_plot_data(self, x, y, labels, samples=500):
        # x_data = self.img[x].values.reshape(-1, 1)
        # y_data = self.img[y].values.reshape(-1, 1)
        # label_data = self.img[labels].values.reshape(-1, 1)

        # plot_data = np.concatenate([x_data, y_data, label_data], dtype='float32', axis=1)

        # # slice for valid_data only
        # invalid = plot_data[:, -1] == -1
        # plot_data = plot_data[~invalid]
        
        # # get some water pts (5%) at least
        # water_data = plot_data[plot_data[:, 2]==self.water_cluster]
        # water_idxs = np.random.randint(len(water_data), size=int(0.05*samples))
        # water_data = water_data[water_idxs]

        # # get a small random sample
        # no_water_data = plot_data[plot_data[:, 2]!=self.water_cluster]
        # no_water_idxs = np.random.randint(len(no_water_data), size=int(0.95*samples))
        # no_water_data = no_water_data[no_water_idxs]

        # plot_data = np.concatenate([water_data, no_water_data], axis=0)

        # create an array with the necessary data and mask it
        data = self.img[[x, y, labels]].to_array()
        data = data.where(~self.img['mask'], other=np.nan)

        # get some water pts (5%)
        water_data = data.where(self.img['watermask'] == 1, other=np.nan).to_numpy().reshape(3, -1).transpose(1, 0)
        water_data = water_data[~np.isnan(water_data).all(axis=1)]
        water_idxs = water_idxs = np.random.randint(len(water_data), size=int(0.05*samples))
        water_data = water_data[water_idxs]

        no_water_data = data.where(self.img[labels] != self.water_cluster, other=np.nan).to_numpy().reshape(3, -1).transpose(1, 0)
        no_water_data = no_water_data[~np.isnan(no_water_data).all(axis=1)]
        no_water_idxs = np.random.randint(len(no_water_data), size=int(0.95*samples))
        no_water_data = no_water_data[no_water_idxs]

        plot_data = np.concatenate([water_data, no_water_data], axis=0)

        return plot_data    
  
    def plot_graphs(self, axes, cols=1, size=7, save_folder=None):
        if 'clusters' not in self.img:
            print('You should run WaterDetect first')
            return 

        # listify the axes
        if isinstance(axes[0], str):
            axes = [axes]

        # create the names for the clusters
        cluster_names = [f'Cluster {i}' for i in range(self.max_k)]

        # create the area for the graphs
        rows = len(axes)//cols + (0 if len(axes) % cols == 0 else 1)
        fig, ax = plt.subplots(
            nrows=rows, 
            ncols=cols, 
            figsize=(size*cols*1.2, size*rows),
            num=1,
            clear=True
        )

        fig.suptitle = self.img_item.id
        
        for i, graph in enumerate(axes):
            plot_data = self.get_plot_data(graph[0], graph[1], 'clusters')

            graph_options = {'x_label': graph[0],
                             'y_label': graph[1]}

            self.plot_clustered_data(data=plot_data,
                                     cluster_names=cluster_names, 
                                     file_name=None,
                                     graph_options=graph_options,    
                                     water_cluster=self.water_cluster,
                                     ax=(ax if len(axes)==1 else ax.reshape(-1)[i])
                                    )
   
        if save_folder is not None:
            fig.savefig(Path(save_folder)/f'{self.img_item.id[:38]}_Graphs.png', dpi=150)
            # fig.clear()
            # plt.close('all')
            # del ax
            # del fig

    # ----------------------------------------------------------------------------------
    # ############################### DUNDER FUNCTIONS ###############################
    def __repr__(self):
        s = f'*** {type(self).__name__} class *** \n'
        s += f'Img loaded with following bands: {self.img.data_vars}\n'
        if 'watermask' not in self.img:
            s += 'Water mask not processed. Call .run_detect_water() method.'
        else:
            s += 'Water mask processed. Access .water_mask or .cluster_matrix for results.'
        return s

    def __del__(self):
        self.img.close()
        del self.img
    
    # https://pccompute.westeurope.cloudapp.azure.com/compute/user/mauricio@ana.gov.br/?token=1d066f20062247ae9ec3684f5ea68def