import cv2
import numpy as np
import scipy as sp


def myGrabCut(class_ims, recs, num_iter):
    result = []
    for i in range(len(class_ims)):
        img = class_ims[i]
        rect = recs[i]
        mask = np.zeros(img.shape[:2],np.uint8)
        bgdModel = np.zeros((1,65),np.float64)
        fgdModel = np.zeros((1,65),np.float64)
        cv2.grabCut(img,mask,rect,bgdModel,fgdModel, num_iter ,cv2.GC_INIT_WITH_RECT)
        mask2 = np.where((mask==2)|(mask==0),1,0).astype('uint8')
        img = img*mask2[:,:,np.newaxis]
        result.append(img)
    return result
