__version__ = '0.0.1'

import os
import logging
from pathlib import Path

logger = logging.getLogger('pc_waterdetect_ini')

# Correct the environment variable for rasterio / Proj.db issue
# https://github.com/rasterio/rasterio/issues/1850
if 'PROJ_LIB' not in os.environ:
    if 'GDAL_DATA' not in os.environ:
        logger.error(f'GDAL_DATA not set in environment variables. Check if GDAL is installed correctly')
    else:
        gdal_data = Path(os.environ['GDAL_DATA'])
        if (gdal_data/'proj').exists():
            os.environ['PROJ_LIB'] = (gdal_data/'proj').as_posix()
        else:
            os.environ['PROJ_LIB'] = (gdal_data.parent/'proj').as_posix()

# Quick testing the proj.db
from rasterio.crs import CRS
crs = CRS.from_epsg(4326)
if crs is not None:
    print('GDAL Proj.db installed correctly')

