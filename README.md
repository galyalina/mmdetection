## This project is a fork of original [mmdetection](https://github.com/open-mmlab/mmdetection) library.
All installation and implementation details can be found in original [README](README.md)

## Configuration for training the network on Toulouse dataset
1. [config/repppoints_toulouse] Folder with configuration files we used during the training of RepPoints network
2. [config/faster_rcnn_toulouse] Folder with configuration files we used during the training of Faster R-CNN network

## Code for preproccessing and generating datasets 

1. [plot_analytics_reppoints.py](tools/plot_analytics_reppoints.py) Script for plotting learning cureves and mAP graphs 
2. [dataset_preprocessing_cropping.py](preprocessing/dataset_preprocessing_cropping.py) Cropping images with sliding window approach, with and without overlap 
3. [dataset_preprocessing_cropping_tiff.py](preprocessing/dataset_preprocessing_cropping_tiff.py)Cropping images with sliding window approach, with and without overlap, preserving geo spatial data
4. [data_annotations_generation_from_segmentation_mask.py](preprocessing/data_annotations_generation_from_segmentation_mask.py) Skript that produces lables from masks files
5. [data_annotations_generation_from_buildings_mask.py](preprocessing/data_annotations_generation_from_buildings_mask.py) Skript that produces lables from masks files
6. [data_augmentation.py](preprocessing/data_augmentation.py) Tool to generate augmentation images
7. [download_osm_segmentation.py](preprocessing/download_osm_segmentation.py) Script to download OSM lables and generate mask files
8. [data_annotations_generation.py](preprocessing/data_annotations_generation.py.py) Script to generate COCO annotations from buildings or segmentation annotations
9. [data_visuzliazation.py](preprocessing/data_visuzliazation.py) Application of COCO JSON file on images in order to visualize labels.

## Citation

If you use this toolbox or benchmark in your research, please cite original project.

```
@article{mmdetection,
  title   = {{MMDetection}: Open MMLab Detection Toolbox and Benchmark},
  author  = {Chen, Kai and Wang, Jiaqi and Pang, Jiangmiao and Cao, Yuhang and
             Xiong, Yu and Li, Xiaoxiao and Sun, Shuyang and Feng, Wansen and
             Liu, Ziwei and Xu, Jiarui and Zhang, Zheng and Cheng, Dazhi and
             Zhu, Chenchen and Cheng, Tianheng and Zhao, Qijie and Li, Buyu and
             Lu, Xin and Zhu, Rui and Wu, Yue and Dai, Jifeng and Wang, Jingdong
             and Shi, Jianping and Ouyang, Wanli and Loy, Chen Change and Lin, Dahua},
  journal= {arXiv preprint arXiv:1906.07155},
  year={2019}
}
```

## Additional libraries
Please cite this library if used
[Lydorn](https://github.com/Lydorn) developed by Nicolas Girard, used in getting OSM lables tool.


## License

This project is released under the [Apache 2.0 license](LICENSE).



