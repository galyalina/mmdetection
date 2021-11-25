# import os
#
# import lydorn_utils.geo_utils as utils
# import numpy as np
# import rasterio
# from PIL import Image, ImageDraw
# from rasterio.windows import Window
#
# # from mmdet.apis import inference_detector, init_detector, show_result_pyplot
#
# PATH = "../../data/toulouse/unlabled"
# DIRECTORY_CROPPED_IMAGE = PATH + "/train/"
# DIRECTORY_MASK_IMAGE = PATH + "/mask/"
#
#
# #
# # def get_prediction(file):
# #     # Choose to use a config and initialize the detector
# #     config = PATH + '/checkpoints/reppoints_moment_r101_fpn_gn-neck+head_2x_toulouse.py'
# #     # Setup a checkpoint file to load
# #     checkpoint = 'checkpoints/epoch_24.pth'
# #     # initialize the detector
# #     model = init_detector(config, checkpoint, device='cuda:0')
# #     result = inference_detector(model, file)
# #     # Let's plot the result
# #     show_result_pyplot(model, file, result, score_thr=0.3)
#
#
# def get_segmentation(image_name):
#     with rasterio.open(DIRECTORY_CROPPED_IMAGE + image_name) as src:
#         mask = Image.new('1', (src.width, src.height), "#000000")
#         mask_draw = ImageDraw.Draw(mask)
#         poligons = utils.get_polygons_from_osm(DIRECTORY_CROPPED_IMAGE + image_name, tag="building", ij_coords=False)
#         for p in poligons:
#             result = list(map(tuple, np.array(p).astype(int)))
#             mask_draw.polygon(result, fill='#ee7621', outline='#ee7621')
#         mask.save(DIRECTORY_MASK_IMAGE + image_name)
#
#
# if __name__ == '__main__':
#     for subdir, dirs, files in os.walk(DIRECTORY_CROPPED_IMAGE):
#         for file in files:
#             if file.lower().endswith('.tif'):
#                 if not os.path.exists(DIRECTORY_MASK_IMAGE + file):
#                     print("get mask for " + file)
#                     get_segmentation(file)
#                     # get_prediction(file)
#                 # else:
#                 # print("mask " + file + " already exists")
