import cv2
import os
from torchvision import transforms
import numpy as np

data_root='/home/wangyike/PST900_RGBT_Dataset/'
split = 'train'
label = os.listdir(data_root+split+'/labels')
os.makedirs(data_root + split + '/binary_labels/',exist_ok=True)
os.makedirs(data_root + split + '/edge/',exist_ok=True)
os.makedirs(data_root + split + '/bound/',exist_ok=True)
for i in label:
    label_path1 = i
    imgrgb= cv2.imread(data_root+split+'/labels/' + label_path1  , 0)



    def tensor_to_PIL(tensor):
        image = tensor.squeeze(0)
        image = unloader(image)
        return image



    x1 = cv2.Sobel(imgrgb, cv2.CV_16S, 1, 0)
    y1 = cv2.Sobel(imgrgb, cv2.CV_16S, 0, 1)


    absX1 = cv2.convertScaleAbs(x1)
    absY1 = cv2.convertScaleAbs(y1)


    dst1 = cv2.addWeighted(absX1, 0.5, absY1, 0.5, 0)

    kernel = np.ones((9, 9), np.float32)
    dst1 = cv2.dilate(dst1, kernel)

    loader = transforms.Compose([
        transforms.ToTensor()])
    unloader = transforms.ToPILImage()



    imgrgb[imgrgb!=0]=255
    cv2.imwrite(data_root + split + '/binary_labels/' + label_path1 + '.png', imgrgb)


    dst1 = loader(dst1)

    c = tensor_to_PIL(dst1)
    c = np.array(c)

    cv2.imwrite(data_root+split+'/edge/' + label_path1 + '.png', c)

    dst = dst1*255

    c = tensor_to_PIL(dst)
    c = np.array(c)

    cv2.imwrite(data_root+split+'/bound/' + label_path1 + '.png', c)


