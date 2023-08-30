import cv2
import os
import numpy as np
import random

if __name__ == "__main__":

    def random_crop(image1, image2, size=512):

        #        image = resize_image(image)

        h, w = image1.shape[:2]

        y = np.random.randint(0, h - size)
        x = np.random.randint(0, w - size)

        data = image1[y:y + size, x:x + size, :]
        gt = image2[y:y + size, x:x + size, :]

        rr = np.random.randint(0, 2)
        if rr == 0:
            data = cv2.flip(data, 0)
            gt = cv2.flip(gt, 0)
            return data, gt
        elif rr == 1:
            #            img_list.append(imgl0)
            data = cv2.flip(data, 1)
            gt = cv2.flip(gt, 1)
            return data, gt
        else:
            #            img_list.append(imgl1)
            return data, gt


    datapath = r"./data3/test_image/"  # change this dirpath.
    listdir = os.listdir(datapath)
    gtpath = r"./data3/test_gt/"  # change this dirpath.

    dnewdir = os.path.join(datapath, 'split2')
    gnewdir = os.path.join(gtpath, 'split2')  # make a new dir in dirpath.

    if (os.path.exists(dnewdir) == False):
        os.mkdir(dnewdir)
    if (os.path.exists(gnewdir) == False):
        os.mkdir(gnewdir)

    for i in listdir:
        if os.path.isdir(os.path.join(datapath, i)):
            continue
        datafilepath = os.path.join(datapath, i)
        gtfilepath = os.path.join(gtpath, i)
        filename = i.split('.')[0]

        img = cv2.imread(datafilepath)
        gtimg = cv2.imread(gtfilepath)
        count = 1
        img_list = []
        gtimg_list = []
        while count <= 150:
            imgg, gtimgg = random_crop(img, gtimg)

            img_path = os.path.join(dnewdir, filename) + "_" + str(count) + ".jpg"
            gtimg_path = os.path.join(gnewdir, filename) + "_" + str(count) + ".jpg"
            cv2.imwrite(img_path, imgg)
            cv2.imwrite(gtimg_path, gtimgg)
            count += 1

