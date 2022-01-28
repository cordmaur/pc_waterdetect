__version__ = '0.0.1'

import os
import logging

logger = logging.getLogger('pc_waterdetect_ini')
# Correct the environment variable for rasterio / Proj.db issue
# https://github.com/rasterio/rasterio/issues/1850
if 'PROJ_LIB' not in os.environ:
    if 'GDAL_DATA' not in os.environ:
        logger.error(f'GDAL_DATA not set in environment variables. Check if GDAL is installed correctly')
    else:
        os.environ['PROJ_LIB'] = os.environ['GDAL_DATA'] + r'\proj'
    