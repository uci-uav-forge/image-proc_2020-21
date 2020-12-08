from PIL import Image 
from PIL.ExifTags import TAGS, GPSTAGS
from osgeo import gdal,ogr,osr
import affine

def get_exif_data(image):
    exif_data = {}
    info = image._getexif()
    if info:
        for tag, value in info.items():
            decoded = TAGS.get(tag, tag)
            if decoded == "GPSInfo":
                gps_data = {}
                for t in value:
                    sub_decoded = GPSTAGS.get(t, t)
                    gps_data[sub_decoded] = value[t]

                exif_data[decoded] = gps_data
            else:
                exif_data[decoded] = value

    return exif_data 

def _get_if_exist(data, key):
    if key in data:
        return data[key]
    else: 
        pass

def get_decimal_coordinates(info):
    for key in ['Latitude', 'Longitude']:
        if 'GPS'+key in info and 'GPS'+key+'Ref' in info:
            e = info['GPS'+key]
            ref = info['GPS'+key+'Ref']
            info[key] = ( e[0][0]/e[0][1] +
                          e[1][0]/e[1][1] / 60 +
                          e[2][0]/e[2][1] / 3600
                        ) * (-1 if ref in ['S','W'] else 1)

    if 'Latitude' in info and 'Longitude' in info:
        return [info['Latitude'], info['Longitude']]

def get_lat_lon(exif_data):
    gps_info = exif_data["GPSInfo"]
    lat = None
    lon = None

    if "GPSInfo" in exif_data:
        gps_info = exif_data["GPSInfo"]

        gps_latitude = _get_if_exist(gps_info, "GPSLatitude")
        gps_latitude_ref = _get_if_exist(gps_info, "GPSLatitudeRef")
        gps_longitude = _get_if_exist(gps_info, "GPSLongitude")
        gps_longitude_ref = _get_if_exist(gps_info, "GPSLongitudeRef")

        if gps_latitude and gps_latitude_ref and gps_longitude and gps_longitude_ref:
            lat, lon = get_decimal_coordinates(gps_info)
            # if gps_latitude_ref != "N":
            #     lat = 0 - lat

            # if gps_longitude_ref != "E":
            #     lon = 0 - lon

        return lat, lon


def ReprojectCoords(coords,src_srs,tgt_srs):
    ''' Reproject a list of x,y coordinates.

        @type geom:     C{tuple/list}
        @param geom:    List of [[x,y],...[x,y]] coordinates
        @type src_srs:  C{osr.SpatialReference}
        @param src_srs: OSR SpatialReference object
        @type tgt_srs:  C{osr.SpatialReference}
        @param tgt_srs: OSR SpatialReference object
        @rtype:         C{tuple/list}
        @return:        List of transformed [[x,y],...[x,y]] coordinates
    '''
    trans_coords=[]
    transform = osr.CoordinateTransformation( src_srs, tgt_srs)
    for x,y in coords:
        x,y,z = transform.TransformPoint(x,y)
        trans_coords.append([x,y])
    return trans_coords


def retrieve_pixel_value(geo_coord, data_source):
    # https://gis.stackexchange.com/questions/221292/retrieve-pixel-value-with-geographic-coordinate-as-input-with-gdal
    """Return floating-point value that corresponds to given point."""
    x, y = geo_coord[0], geo_coord[1]
    forward_transform = affine.Affine.from_gdal(*data_source.GetGeoTransform())
    reverse_transform = ~forward_transform
    px, py = reverse_transform * (x, y)
    px, py = int(px + 0.5), int(py + 0.5)
    pixel_coord = px, py
    return pixel_coord


if __name__ == "__main__":
    '''starting point to create image masks for output map for continuously expanding map'''
    # get center gps coords of new image
    image = Image.open("image-proc_2020-21/Mapping/images/DJI_0025.JPG")
    exif_data = get_exif_data(image)
    lat, lon = get_lat_lon(exif_data)
    print(lat, lon)

    # get projection data from tif file
    raster='image-proc_2020-21/odm_orthophoto/odm_orthophoto.tif'
    ds=gdal.Open(raster)
    src_srs=osr.SpatialReference()
    src_srs.ImportFromWkt(ds.GetProjection())
    tgt_srs = src_srs.CloneGeogCS()

    # convert gps to utm
    utm = latlon=ReprojectCoords([(lat, lon)],tgt_srs,src_srs)
    print(utm)

    # convert utm to pixel coords on tif image 
    pixel_coord_x, pixel_coord_y = retrieve_pixel_value(utm[0], ds)
    print(pixel_coord_x, pixel_coord_y)