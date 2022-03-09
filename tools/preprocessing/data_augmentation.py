import os

import cv2
import mxnet as mx
from mxnet.gluon.data.vision import transforms

PATH = "../../data/toulouse/model_data/train"
# DIRECTORY_ANNOTATIONS = PATH + "/annotations/"
DIRECTORY_IMAGE = PATH + "/images/"
DIRECTORY_MASK = PATH + "/masks/"
NUM_OF_AUGMENTED_IMAGE = 5


def delete_folder(path):
    if not os.path.exists(path):
        os.makedirs(path)
    for root, dirs, files in os.walk(path, topdown=False):
        for name in files:
            try:
                os.remove(os.path.join(root, name))
            except Exception:
                print()
        print()
        for name in dirs:
            try:
                os.rmdir(os.path.join(root, name))
            except Exception:
                print()


if __name__ == '__main__':
    # delete_folder(DIRECTORY_AUGMENTATION)
    # if not os.path.exists(DIRECTORY_AUGMENTATION):
    #     os.makedirs(DIRECTORY_AUGMENTATION)

    for subdir, dirs, files in os.walk(DIRECTORY_IMAGE):
        for index, file in enumerate(files, start=1):
            if file == '.DS_Store':
                continue
            mask = cv2.imread(DIRECTORY_MASK + file)
            image = mx.image.imread(subdir + file, flag=1, to_rgb=False)
            transform = transforms.Compose([
                transforms.RandomSaturation(saturation=0.5),
                transforms.RandomLighting(alpha=0.2),
                transforms.RandomHue(hue=0.1),
                transforms.RandomContrast(contrast=0.2)
            ])

            transformed = [transform(image) for _ in range(NUM_OF_AUGMENTED_IMAGE)]
            for i in range(NUM_OF_AUGMENTED_IMAGE):
                file_name = file + "_augmented_" + str(i)
                cv2.imwrite(DIRECTORY_MASK + file_name + '.tif', mask)
                cv2.imwrite(DIRECTORY_IMAGE + file_name + ".tif",
                            transformed[i].asnumpy())
