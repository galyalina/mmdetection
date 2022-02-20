import os

import lydorn_utils.geo_utils as utils
import numpy as np
import rasterio
from PIL import Image, ImageDraw
from rasterio.windows import Window

PATH = "../../data/zeven"
DIRECTORY_OUTPUT = PATH + "/large_osm_polygons/"
DIRECTORY_INPUT = PATH + "/large_test/"
IMAGE_SIZE = 460
IMAGE_OVERLAP_PERCENTAGE = 0


def start_points(size, split_size, overlap=0):
    points = [(0, min(size, split_size))]
    stride = int(split_size * (1 - overlap))
    counter = 1
    while True:
        pt = stride * counter
        if pt + split_size >= size:
            points.append((pt, size - pt))
            break
        else:
            points.append((pt, split_size))
        counter += 1
    return points


def get_segmentation_by_regions(image_name):
    with rasterio.open(DIRECTORY_INPUT + image_name) as src:
        X_points = start_points(src.height, IMAGE_SIZE, IMAGE_OVERLAP_PERCENTAGE)
        Y_points = start_points(src.width, IMAGE_SIZE, IMAGE_OVERLAP_PERCENTAGE)
        index = 0
        for y, h in Y_points:
            for x, w in X_points:

                window = Window(x, y, min(src.width - x, IMAGE_SIZE), min(src.height - y, IMAGE_SIZE))
                kwargs = src.meta.copy()
                kwargs.update({
                    'height': window.height,
                    'width': window.width,
                    'transform': rasterio.windows.transform(window, src.transform)})

                input_image = DIRECTORY_OUTPUT + "cropped_" + str(x) + "_" + str(y) + ".tiff"
                with rasterio.open(input_image, 'w', **kwargs) as dst:
                    dst.write(src.read(window=window))
                image_to_show = Image.open(input_image)
                img_draw = ImageDraw.Draw(image_to_show)
                str_proj_EPSG_25832 = "PROJCS[\"ETRS89 / UTM zone 32N\",GEOGCS[\"ETRS89\",DATUM[\"European_Terrestrial_Reference_System_1989\",SPHEROID[\"GRS 1980\",6378137,298.257222101,AUTHORITY[\"EPSG\",\"7019\"]],TOWGS84[0,0,0,0,0,0,0],AUTHORITY[\"EPSG\",\"6258\"]],PRIMEM[\"Greenwich\",0,AUTHORITY[\"EPSG\",\"8901\"]],UNIT[\"degree\",0.0174532925199433,AUTHORITY[\"EPSG\",\"9122\"]],AUTHORITY[\"EPSG\",\"4258\"]],PROJECTION[\"Transverse_Mercator\"],PARAMETER[\"latitude_of_origin\",0],PARAMETER[\"central_meridian\",9],PARAMETER[\"scale_factor\",0.9996],PARAMETER[\"false_easting\",500000],PARAMETER[\"false_northing\",0],UNIT[\"metre\",1,AUTHORITY[\"EPSG\",\"9001\"]],AXIS[\"Easting\",EAST],AXIS[\"Northing\",NORTH],AUTHORITY[\"EPSG\",\"25832\"]]"
                poligons = utils.get_polygons_from_osm(input_image,
                                                       tag="building",
                                                       ij_coords=False,
                                                       specific_projection=str_proj_EPSG_25832)
                for p in poligons:
                    result = list(map(tuple, np.array(p).astype(int)))
                    img_draw.polygon(result, outline='#00ff00')
                    bbox = p[:, 0].min(), p[:, 0].max(), p[:, 1].min(), p[:, 1].max()
                    img_draw.rectangle([(bbox[0], bbox[2]), (bbox[1], bbox[3])], outline='#00ff00', width=2)
                    image_to_show.save(input_image)
                index += 1
                # test one image
                return


def get_segmentation_one_chunk(image_name):
    mask = Image.open(DIRECTORY_INPUT + image_name)
    mask_draw = ImageDraw.Draw(mask)
    str_proj_EPSG_25832 = "PROJCS[\"ETRS89 / UTM zone 32N\",GEOGCS[\"ETRS89\",DATUM[\"European_Terrestrial_Reference_System_1989\",SPHEROID[\"GRS 1980\",6378137,298.257222101,AUTHORITY[\"EPSG\",\"7019\"]],TOWGS84[0,0,0,0,0,0,0],AUTHORITY[\"EPSG\",\"6258\"]],PRIMEM[\"Greenwich\",0,AUTHORITY[\"EPSG\",\"8901\"]],UNIT[\"degree\",0.0174532925199433,AUTHORITY[\"EPSG\",\"9122\"]],AUTHORITY[\"EPSG\",\"4258\"]],PROJECTION[\"Transverse_Mercator\"],PARAMETER[\"latitude_of_origin\",0],PARAMETER[\"central_meridian\",9],PARAMETER[\"scale_factor\",0.9996],PARAMETER[\"false_easting\",500000],PARAMETER[\"false_northing\",0],UNIT[\"metre\",1,AUTHORITY[\"EPSG\",\"9001\"]],AXIS[\"Easting\",EAST],AXIS[\"Northing\",NORTH],AUTHORITY[\"EPSG\",\"25832\"]]"
    poligons = utils.get_polygons_from_osm(DIRECTORY_INPUT + image_name,
                                           tag="building",
                                           ij_coords=False,
                                           specific_projection=str_proj_EPSG_25832)
    for p in poligons:
        result = list(map(tuple, np.array(p).astype(int)))
        mask_draw.polygon(result, outline='#ee7621')
        bbox = p[:, 0].min(), p[:, 0].max(), p[:, 1].min(), p[:, 1].max()
        mask_draw.rectangle([(bbox[0], bbox[2]), (bbox[1], bbox[3])],
                            outline='#ff00ff', width=3)
    mask.save(DIRECTORY_OUTPUT + image_name)


if __name__ == '__main__':
    for subdir, dirs, files in os.walk(DIRECTORY_INPUT):
        for file in files:
            if file.lower().endswith('.tif'):
                # get_segmentation_one_chunk(file)
                get_segmentation_by_regions(file)


def rotateImage(image, angle, center=None, scale=1.0):
    (h, w) = image.shape[:2]
    if center is None:
        center = (w / 2, h / 2)
    # Perform the rotation
    M = cv2.getRotationMatrix2D(center, angle, scale)
    rotated = cv2.warpAffine(image, M, (w, h))
    return rotated


# stores mouse position in global variables ix(for x coordinate) and iy(for y coordinate)
# on double click inside the image
def select_point(event, x, y, flags, param):
    global ix, iy
    if event == cv2.EVENT_LBUTTONDBLCLK:  # captures left button double-click
        ix, iy = x, y

def check_boxes:
    img = cv2.imread('sample.jpg')
    cv2.namedWindow('image')
    # bind select_point function to a window that will capture the mouse click
    cv2.setMouseCallback('image', select_point)
    cv2.imshow('image', img)
    k = cv2.waitKey(0) & 0xFF
    if k == ord('a'):
        # print(k)
        # print(ix, iy)
        rotated_img = rotateImage(img, 45, (ix, iy))
        cv2.imshow('rotated', rotated_img)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
