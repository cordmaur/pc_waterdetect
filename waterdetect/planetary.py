from geojson import Point
from pystac_client import Client
from math import ceil
import rioxarray as xrio
from pathlib import Path
import planetary_computer as pc
import numpy as np

import matplotlib.pyplot as plt

# ----------------------------------------------------------------
# ************** SEARCH FUNCTIONS ********************
# ----------------------------------------------------------------
def search_tiles(tile, date_range, max_clouds=90, reverse=False):
    '''Return a list of PYSTAC Items corresponding to the tile (without T) and the date_range specified'''
    
    # Create the query
    query = {
        "s2:mgrs_tile": {
          "eq": tile
        },
        "eo:cloud_cover": {
            "lt": max_clouds
        }
      }
    
    catalog = Client.open("https://planetarycomputer.microsoft.com/api/stac/v1")
    search = catalog.search(collections=["sentinel-2-l2a"], query=query, datetime=date_range)
    tiles = search.get_all_items().items
    
    return sorted(tiles, key=lambda x: x.datetime, reverse=reverse)
    

def search_img(coords, date):
    '''Return the image(s) in a coordinate (lon, lat) in a specific date'''

    # Open Sentinel 2 Catalog
    catalog = Client.open("https://planetarycomputer.microsoft.com/api/stac/v1")

    # Create a point GeoJson with the coords and perform a search
    pt = Point(coords)
    search = catalog.search(collections=["sentinel-2-l2a"], 
                        datetime=date,
                        intersects=pt)
    
    # return the items
    return search.get_all_items().items


# ----------------------------------------------------------------
# ************** PLOTTING FUNCTIONS ********************
# ----------------------------------------------------------------
def plot_stac_asset(asset, ax=None, **kwargs):
    """
    Plot a single stac item to an existing ax or a new one (if ax = None)
    Kwargs will be passed to imshow
    """
    
    plt.style.use('default')

    if ax is None:
        _, ax = plt.subplots(**kwargs)
    
    href = Path(asset.href)
    
    if href.suffix == '.tif':
        # open the item and downscale by 10x 
        array = xrio.open_rasterio(href, masked=True).squeeze()[::10, ::10]
        
    else:
        # open the image as png or jpg, etc.
        array = plt.imread(href)
    
    # display the resulting array in axis ax
    ax.imshow(array)


def plot_previews(items, asset='preview', cols=3, base_size=5):
    
    plt.style.use('default')

    # number of items
    n = len(items)
    
    rows = ceil(n / cols)
    
    _, axs = plt.subplots(rows, cols, figsize=(cols * base_size, rows * base_size))
    
    # iterate through the items to display them in lower resolution
    for i, item in enumerate(items):
        
        ax = axs.reshape(-1)[i] if isinstance(axs, np.ndarray) else axs

        # get the signed href
        signed_href = pc.sign(item.assets[asset].href)

        # open the image in the href
        img = xrio.open_rasterio(signed_href)
        ax.imshow(img.values.transpose(1, 2, 0))
        
        # plot a title
        title = item.id[:38]
        ax.set_title(title)

        
    
def plot_stac_items(items, asset_name, cols=3, base_size=5):
    
    plt.style.use('default')

    # number of items
    n = len(items)
    
    rows = ceil(n / cols)
    
    _, axs = plt.subplots(rows, cols, figsize=(cols * base_size, rows * base_size))
    
    # iterate through the items to display them in lower resolution
    for i, item in enumerate(items):
        
        ax = axs.reshape(-1)[i] if isinstance(axs, np.ndarray) else axs

        plot_stac_asset(item.assets[asset_name], ax)
        
        # array = xrio.open_rasterio(item.assets[asset_name].href, masked=True).squeeze()
        # mean = array.mean().values
        # total = array.sum().values
        
        title = item.id
        # title = title + f'\nMean:{mean:.3f} Total: {total/10000}'
        ax.set_title(title)
        # open the monthly watermask


