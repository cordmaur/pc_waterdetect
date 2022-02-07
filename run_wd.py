import ee
from waterdetect.automation import process_period
import argparse


def main(tile=None, period=None):
    """
    Main function when called from the console.
    In this version, just tile and period will be configured by arguments.
    Output folder, and logging folder are hard coded.
    """
    
    ee.Initialize()
    
    cluster_bands = ['mndwi', 'ndwi', 'B12']
    output_folder = 'E:/output/planetary'
    
    process_period(tile, period, output_folder, cluster_bands)

def test():
    print('testou')
    
# check if this file has been called as script    
if __name__ == '__main__':   
    
    # Parse the arguments before calling MAIN
    parser = argparse.ArgumentParser()
    
    parser.add_argument("-t", "--Tile", help='Sentinel 2 Tile to be processed', required=True, type=str)
    parser.add_argument("-p", "--Period", help='Time period', required=True, type=str)
    
    args = parser.parse_args()
    
    main(tile=args.Tile, period=args.Period)
