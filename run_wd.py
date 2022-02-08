import ee
from waterdetect.automation import process_period, test_memory
import argparse

# import os
# os.environ['PROJ_LIB'] = r'D:\Programs\miniconda3\envs\rasterio\Library\share\proj'
# os.environ['GDAL_DATA'] = r'D:\Programs\miniconda3\envs\rasterio\Library\share'

def main(tile=None, period=None, mt=False):
    """
    Main function when called from the console.
    In this version, just tile and period will be configured by arguments.
    Output folder, and logging folder are hard coded.
    """
    
    ee.Initialize()
    
    cluster_bands = ['mndwi', 'ndwi', 'B12']
    output_folder = 'E:/output/planetary'

    if not mt:
        process_period(tile, period, output_folder, cluster_bands)

    else:
        test_memory(tile, period, output_folder, cluster_bands)


# check if this file has been called as script    
if __name__ == '__main__':   
    
    # Parse the arguments before calling MAIN
    parser = argparse.ArgumentParser()
    
    parser.add_argument("-t", "--Tile", help='Sentinel 2 Tile to be processed', required=True, type=str)
    parser.add_argument("-p", "--Period", help='Time period', required=True, type=str)
    parser.add_argument("-m", "--MemoryTest", help='Should be called to make 5 repetitions for memory profiling.', 
                        required=False, action='store_true')
    
    args = parser.parse_args()
    
    main(tile=args.Tile, period=args.Period, mt=args.MemoryTest)


