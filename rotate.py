import cv2
import numpy as np
import matplotlib.pyplot as plt

def rotate_image(image, angle):
    sizeX = image.shape[1]
    sizeY = image.shape[0]
    outputX = np.abs(np.cos(angle/180.0*np.pi))*sizeX + np.abs(np.sin(angle/180.0*np.pi))*sizeY
    outputY = np.abs(np.sin(angle/180.0*np.pi))*sizeX + np.abs(np.cos(angle/180.0*np.pi))*sizeY
    # Fill the boader with transparent pixels
    output = cv2.copyMakeBorder(image, int((outputY - sizeY)/2)+1, int((outputY - sizeY)/2)+1, int((outputX - sizeX)/2)+1, int((outputX - sizeX)/2)+1, cv2.BORDER_CONSTANT, value=[0,0,0,0])
    # Rotate the image
    M = cv2.getRotationMatrix2D((outputX/2, outputY/2), angle, 1)
    rotated = cv2.warpAffine(output, M, (int(outputX), int(outputY)), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0,0))
    return rotated

def rotate_back(image, angle):
    # Resize back to original dimensions
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    originX = np.cos(angle/180.0*np.pi)*w + np.sin(angle/180.0*np.pi)*h
    originY = np.sin(angle/180.0*np.pi)*w + np.cos(angle/180.0*np.pi)*h
    M = cv2.getRotationMatrix2D(center, -angle, 1.0)
    M[0,2] += (w - originX) / 2
    M[1,2] += (h - originY) / 2
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    return rotated