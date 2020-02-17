import numpy as np
import cv2
import pickle

channels = 16 #number of electrodes

images=['data/LFP_0.jpg','data/LFP_375.jpg',
        'data/iCSD_0.jpg','data/iCSD_375.jpg',]

scale_factors=[]

for i in images:
    scale = cv2.imread('data/scale.jpg')
    scale_rgb=cv2.cvtColor(scale.copy(),cv2.COLOR_BGR2RGB)

    img = cv2.imread(i)
    img_rgb = cv2.cvtColor(img.copy(),cv2.COLOR_BGR2RGB)

    found=False
    counter=0

    # print('måten å finne skaleringen er grusom')
    low =[]
    for z in range(0,int(scale_rgb.shape[1]/4)):
        scale_ = scale_rgb[0][z]
        for i in range(img_rgb.shape[0]):
            for j in range(img_rgb.shape[1]):
                if np.all(img_rgb[i][j]==scale_):
                    low.append(z)
                    found=True
                    break
            else:
                continue
            break
        else:
            continue
        break
    if found !=True:
        print('something went wrong when finding scale.')

    high = []
    for z in reversed(range(int(2*scale_rgb.shape[1]/4),scale_rgb.shape[1])):
        scale_ = scale_rgb[0][z]
        for i in range(img_rgb.shape[0]):
            for j in range(img_rgb.shape[1]):
                if np.all(img_rgb[i][j]==scale_):
                    high.append(z)
                    found=True
                    break
            else:
                continue
            break
        else:
            continue
        break
    if found !=True:
        print('something went wrong when finding scale.')

    scale_mv = np.linspace(-5,5,scale_rgb.shape[1])
    scale_factors.append([scale_mv[low],scale_mv[high]])
    # print(scale_mv[low],scale_mv[high])
pickle.dump(scale_factors,open('data/scale_factors.p','wb'))