import io_image 
import numpy as np
import matplotlib.pyplot as plt

img1 = io_image.read_image("data/images/corridor.jpg")
img2 = io_image.read_image("data/images/person.jpg")

def sum_img(img1, img2, a):
    #Alpha debe de valer de entre 0 a 1
    img1 = img1.astype(np.float32)
    img2 = img2.astype(np.float32)
    img_total = (1-a)*img1 + a*img2
    # img_total =  img_total.astype(np.uint8)
    img_total =  img_total.clip(0,255)
    return img_total

def res_img(img1, img2):
    img1 = img1.astype(np.float32)
    img2 = img2.astype(np.float32)
    img_total = img1 - img2
    img_total = img_total.clip(0,255)
    return img_total

def mult_img(img1, img2):
    img1 = img1.astype(np.float32)
    img2 = img2.astype(np.float32)
    img_total = img1 * img2
    img_total=img_total.clip(0,255)
    return img_total

def divi_img(img1, img2):
    img1 = img1.astype(np.float32)
    img2 = img2.astype(np.float32)
    img_total = img1 / (img2 + 0.000001)
    img_total = img_total.clip(0,255)
    return img_total

mascara = mult_img(img1, img2)
recorte = res_img(img1,mascara)
imagen_sumada = sum_img(recorte, img2, 0.5)
plt.imshow(imagen_sumada, cmap='gray')
plt.show()