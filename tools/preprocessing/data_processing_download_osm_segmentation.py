import os

import lydorn_utils.geo_utils as utils
import numpy as np
import rasterio
from PIL import Image, ImageDraw
from rasterio.windows import Window

PATH = "../../data/zeven"
DIRECTORY_CROPPED_IMAGE = PATH + "/train/"
DIRECTORY_MASK_IMAGE = PATH + "/mask/"


# def generate_new_image(src, image_path):
# with rasterio.open(src) as src:
#     window = Window(0, 0, width=src.width, height=src.height)
#     kwargs = src.meta.copy()
#     kwargs.update({
#         'height': window.height,
#         'width': window.width,
#         'transform': rasterio.windows.transform(window, src.transform)})
#
#     with rasterio.open(image_path, 'w', **kwargs) as dst:
#         dst.write()
#         return dst


def get_segmentation(image_name):
    with rasterio.open(DIRECTORY_CROPPED_IMAGE + image_name) as src:
        mask = Image.new('1', (src.width, src.height), "#000000")
        mask_draw = ImageDraw.Draw(mask)
        str_proj_EPSG_25832 = "PROJCS[\"ETRS89 / UTM zone 32N\",GEOGCS[\"ETRS89\",DATUM[\"European_Terrestrial_Reference_System_1989\",SPHEROID[\"GRS 1980\",6378137,298.257222101,AUTHORITY[\"EPSG\",\"7019\"]],TOWGS84[0,0,0,0,0,0,0],AUTHORITY[\"EPSG\",\"6258\"]],PRIMEM[\"Greenwich\",0,AUTHORITY[\"EPSG\",\"8901\"]],UNIT[\"degree\",0.0174532925199433,AUTHORITY[\"EPSG\",\"9122\"]],AUTHORITY[\"EPSG\",\"4258\"]],PROJECTION[\"Transverse_Mercator\"],PARAMETER[\"latitude_of_origin\",0],PARAMETER[\"central_meridian\",9],PARAMETER[\"scale_factor\",0.9996],PARAMETER[\"false_easting\",500000],PARAMETER[\"false_northing\",0],UNIT[\"metre\",1,AUTHORITY[\"EPSG\",\"9001\"]],AXIS[\"Easting\",EAST],AXIS[\"Northing\",NORTH],AUTHORITY[\"EPSG\",\"25832\"]]"
        poligons = utils.get_polygons_from_osm(DIRECTORY_CROPPED_IMAGE + image_name,
                                               tag="building",
                                               ij_coords=False,
                                               specific_projection=str_proj_EPSG_25832)
        for p in poligons:
            result = list(map(tuple, np.array(p).astype(int)))
            mask_draw.polygon(result, fill='#ee7621', outline='#ee7621')
        mask.save(DIRECTORY_MASK_IMAGE + image_name)


if __name__ == '__main__':
    for subdir, dirs, files in os.walk(DIRECTORY_CROPPED_IMAGE):
        for file in files:
            if file.lower().endswith('.tif'):
                if not os.path.exists(DIRECTORY_MASK_IMAGE + file):
                    print("get mask for " + file)
                    get_segmentation(file)
                # else:
                # print("mask " + file + " already exists")
