import lydorn_utils.geo_utils as utils
import osgeo
import rasterio
from rasterio.windows import Window
import numpy as np
from PIL import Image, ImageDraw
import json


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)


if __name__ == '__main__':
    print(osgeo.gdal.__version__)

    with rasterio.open('../../data/zeven/large/reduced_zeven_32_518_5902_2.tif') as src:
        window = Window(0, 0, 200, 200)
        kwargs = src.meta.copy()
        kwargs.update({
            'height': window.height,
            'width': window.width,
            'transform': rasterio.windows.transform(window, src.transform)})

        with rasterio.open('../../data/toulouse/unlabled/test/cropped.tif', 'w', **kwargs) as dst:
            dst.write(src.read(window=window))

    image_to_show = Image.open('../../data/toulouse/unlabled/test/cropped.tif')
    img_draw = ImageDraw.Draw(image_to_show)

    str_proj_EPSG_25832 = "PROJCS[\"ETRS89 / UTM zone 32N\",GEOGCS[\"ETRS89\",DATUM[\"European_Terrestrial_Reference_System_1989\",SPHEROID[\"GRS 1980\",6378137,298.257222101,AUTHORITY[\"EPSG\",\"7019\"]],TOWGS84[0,0,0,0,0,0,0],AUTHORITY[\"EPSG\",\"6258\"]],PRIMEM[\"Greenwich\",0,AUTHORITY[\"EPSG\",\"8901\"]],UNIT[\"degree\",0.0174532925199433,AUTHORITY[\"EPSG\",\"9122\"]],AUTHORITY[\"EPSG\",\"4258\"]],PROJECTION[\"Transverse_Mercator\"],PARAMETER[\"latitude_of_origin\",0],PARAMETER[\"central_meridian\",9],PARAMETER[\"scale_factor\",0.9996],PARAMETER[\"false_easting\",500000],PARAMETER[\"false_northing\",0],UNIT[\"metre\",1,AUTHORITY[\"EPSG\",\"9001\"]],AXIS[\"Easting\",EAST],AXIS[\"Northing\",NORTH],AUTHORITY[\"EPSG\",\"25832\"]]"
    poligons = utils.get_polygons_from_osm("../../data/toulouse/unlabled/test/cropped.tif",
                                           tag="building",
                                           ij_coords=False,
                                           specific_projection=str_proj_EPSG_25832)
    # specific_projection=str_proj_EPSG_25832)
    for p in poligons:
        result = list(map(tuple, np.array(p).astype(int)))
        print(result)
        with open('../../data/toulouse/unlabled/osm_lables/cropped.json', 'w') as outfile:
            outfile.write(json.dumps(result, cls=NpEncoder))
        img_draw.polygon(result, outline='#ee7621')
        image_to_show.save("../../data/toulouse/unlabled/test/cropped_poligons.tif")
