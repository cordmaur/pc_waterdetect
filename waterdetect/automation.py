import gc
import logging
from pathlib import Path
import profile
from memory_profiler import profile

from .planetary import search_tiles
from .engine import WaterDetect
from .cloudless import get_gee_img


def create_logger(tile):
    # Create a logger for the tile
    logger = logging.getLogger('automation')
    handler = logging.FileHandler(f'./tmp/log_{tile}.txt', mode='a+')
    handler.setFormatter(logging.Formatter("%(asctime)-15s %(levelname)-8s %(message)s"))
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)    
    return logger

# fp = open('memory_profiling.log', 'w+')

def process_img(img, out_folder, cluster_bands, logger, n_jobs=4, retries=3):
    """
    Fully waterdetect one img (stac item) and save the outputs to the out_folder.
    If something goes wrong in the meantime, it will retry the execution.
    """
    
    logger.setLevel(logging.INFO)
    logger.info('*'*50)
    logger.info(f'Processing tile {img}')
    
    while retries > 0:
        try:
            wd = WaterDetect(img, cluster_bands, s2clouds=True, n_jobs=n_jobs)
            wd.run_detect_water()

            # save thumbnails
            wd.plot_thumbs(cols=3, thumbs=['rgb', 'watermask', 'mndwi', 'mask', 'glint', 'ndwi'],  save_folder=out_folder)

            # save graphs
            wd.plot_graphs([['ndwi', 'mndwi'], ['B12', 'mndwi']], cols=2, save_folder=out_folder)

            # save the watermask
            wd.save_geotiff('nodata_watermask', out_folder/f'{wd.img_item.id[:38]}_watermask.tif')

            retries = 0
            logger.setLevel(logging.INFO)
            logger.info(f'{img} - OK')

        except Exception as e:
            
            logger.error(f'Exception {e} when processing img {img}')
            logger.error(f'Releasing memory')
            
            
            retries -= 1
            logger.error(f'There are {retries} retries left!')

        # Regardless the result of the processing, will release the memory by killing wd
        try:
            del wd
        except:
            pass
        
        # And collect the garbage
        gc.collect()
        

def process_period(tile, period, output_folder, cluster_bands):
    
    # Create the output folder (tile name)
    out_path = Path(output_folder)/tile
    if not out_path.exists():
        out_path.mkdir(exist_ok=True)

    imgs = search_tiles(tile, period, reverse=False)
    
    logger = create_logger(tile)
    
    for img in imgs:
        process_img(img, out_path, cluster_bands, logger)


# This function has been created exclusively for profiling the memory 
# @profile(stream=fp)
def memory_test(img, cluster_bands):
        print('MEMORY at beginning')

        out_folder = Path('d:/temp')

        wd = WaterDetect(img, cluster_bands, s2clouds=True, n_jobs=4)
        gc.collect()
        print('MEMORY after loading beginning')

        wd.run_detect_water()
        gc.collect()
        print('MEMORY after running')

        # save thumbnails
        wd.plot_thumbs(cols=3, thumbs=['rgb', 'watermask', 'mndwi', 'mask', 'glint', 'ndwi'],  save_folder=out_folder)
        gc.collect()

        # save graphs
        wd.plot_graphs([['ndwi', 'mndwi'], ['B12', 'mndwi']], cols=2, save_folder=out_folder)
        gc.collect()

        # save the watermask
        wd.save_geotiff('nodata_watermask', out_folder/f'{wd.img_item.id[:38]}_watermask.tif')
        gc.collect()

        del wd
        gc.collect()

        print('MEMORY after releasing the object')


def test_memory(tile, period, output_folder, cluster_bands):
    
    # Create the output folder (tile name)
    out_path = Path(output_folder)/tile
    if not out_path.exists():
        out_path.mkdir(exist_ok=True)

    imgs = search_tiles(tile, period, reverse=False)
    
    logger = create_logger(tile)
    
    # wd = WaterDetect(imgs[0], cluster_bands, s2clouds=False, n_jobs=4)
    # wd.run_detect_water()

    gc.collect()

    for img in imgs[:5]:
        memory_test(img, cluster_bands)


        # process_img(imgs[0], out_path, cluster_bands, logger)

    
