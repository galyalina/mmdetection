import os
import argparse
import cv2
import mxnet as mx
from mxnet.gluon.data.vision import transforms

NUM_OF_AUGMENTED_IMAGE = 10

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("file")
    args = parser.parse_args()
    original_image = args.file
    print(args.file)
    augmented = original_image.split('.')[0]
    mask = cv2.imread(original_image)
    image = mx.image.imread(original_image, flag=1, to_rgb=False)
    transform = transforms.Compose([
        transforms.RandomSaturation(saturation=0.5),
        transforms.RandomLighting(alpha=0.2),
        transforms.RandomHue(hue=0.1),
        transforms.RandomContrast(contrast=0.2)
    ])

    transformed = [transform(image) for _ in range(NUM_OF_AUGMENTED_IMAGE)]
    for i in range(NUM_OF_AUGMENTED_IMAGE):
        file_name = augmented + "_augmented_" + str(i)
        cv2.imwrite(file_name + '.tif', mask)
        cv2.imwrite(file_name + ".tif",
                    transformed[i].asnumpy())
