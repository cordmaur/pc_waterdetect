# This module is responsible for locating and downloading a cloud cover from GEE
import ee
import logging

logger = logging.getLogger('Cloudless')


def get_center(bbox):
    """
    Given a bounding box [lon, lat, lon, lat], return the centroid as [lon, lat]
    """
    lon = (bbox[0] + bbox[2]) / 2
    lat = (bbox[1] + bbox[3]) / 2
    
    return [lon, lat]


def get_gee_img(stac_item, gee_catalog='COPERNICUS/S2_CLOUD_PROBABILITY'):
    """Given a stac item on MS Planetary, get the corresponding gee image"""
    
    # first, we will get the center point of the stac item
    center = ee.Geometry.Point(get_center(stac_item.bbox))
    
    # set start and end dates
    start = stac_item.datetime.strftime('%Y-%m-%dT00:00')
    end = stac_item.datetime.strftime('%Y-%m-%dT23:59')
    
    gee_img = ee.ImageCollection(gee_catalog).filterDate(start, end).filterBounds(center)
    
    # For additionall guarantee, we will extract the time for the ID
    timeid = stac_item.datetime.strftime('%Y%m%dT%H%M%S')
    
    gee_img = gee_img.filterMetadata('system:index', "contains", timeid)
    
    count = gee_img.toList(5).length().getInfo()
    
    if count == 0:
        logger.error(f'{count} image(s) found in collection {gee_catalog}')
        return None
    elif count > 1:
        logger.warning(f'{count} image(s) found in collection {gee_catalog}. Getting the first!!')
        return gee_img.first()
        
    else:
        return gee_img.first()
    

def project_shadows(img, clouds, remove_water_shadow=False, nir_thresh=0.15, proj_distance=1):
    # Identify water pixels from the SCL band.
    not_water = img.select('SCL').neq(6)

    # the dark pixels depend if we are removing or not the water shadows
    if remove_water_shadow:
        dark_pixels = img.select('B8').lt(nir_thresh*1e4).rename('dark_pixels')
    else:
        dark_pixels = img.select('B8').lt(nir_thresh*1e4).multiply(not_water).rename('dark_pixels')

    # Determine the direction to project cloud shadow from clouds (assumes UTM projection).
    shadow_azimuth = ee.Number(90).subtract(ee.Number(img.get('MEAN_SOLAR_AZIMUTH_ANGLE')));

    # Project shadows from clouds for the distance specified by the CLD_PRJ_DIST input.
    cld_proj = (clouds.select('clouds').directionalDistanceTransform(shadow_azimuth, proj_distance*10)
        .reproject(**{'crs': img.select(10).projection(), 'scale': 100})
        .select('distance')
        .mask()
        .rename('cloud_transform'))

    # Identify the intersection of dark pixels with cloud shadow projection.
    shadows = cld_proj.multiply(dark_pixels).rename('shadows')

    
    # Add dark pixels, cloud projection, and identified shadows as image bands.
    # return img.addBands(ee.Image([dark_pixels, cld_proj, shadows]))
    return shadows


def create_cloud_mask(img: ee.Image, cloud_prob: ee.Image, cloud_thresh=40, nir_thresh=0.1, dilation=50):
    """Given a Sentinel 2 image and a S2Cloud probability map, return a cloud/shadow mask"""

    buffer = 50
    
    # create a clouds layer 
    clouds = cloud_prob.select('probability').gt(cloud_thresh).rename('clouds')
    
    # create a shadows layert by projecting the clouds according to solar azimuth angle
    shadows = project_shadows(img=img, 
                              clouds=clouds,
                              remove_water_shadow=True,
                              nir_thresh=nir_thresh,
                              proj_distance=1
                             )
    
    # combine both masks into one
    mask = clouds.select('clouds').add(shadows.select('shadows')).gt(0)

    # Dilate the final mask
    mask = (mask.focalMax(dilation*2/20)
        .reproject(**{'crs': img.select([0]).projection(), 'scale': 20})
        .rename('cloudmask'))
    
    mask = mask.reproject(**{'crs': img.select([10]).projection()})
    
    return mask