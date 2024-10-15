from utils import io_image
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

# mascara = mult_img(img1, img2)
# recorte = res_img(img1,mascara)
# imagen_sumada = sum_img(recorte, img2, 0.5)
# plt.imshow(imagen_sumada, cmap='gray')
# plt.show()


def agrega_ruido_gaussiano(img, sigma):
    h, w = img.shape
    img_ruido = np.random.normal(0,sigma,(h,w))
    img_out = img + img_ruido
    return img_out

def agregar_ruido_multiplicativo(img, sigma):
    w, h = img.shape
    img_ruido = np.random.normal(1,sigma,(w,h))
    img_out = img * img_ruido
    return img_out

def ruido_sal_pimienta(img, p):
    h, w = img.shape
    mat_proba = np.random.random((h,w))
    ruido = np.random.randint(0,2,(h,w))*255
    img_out = np.zeros((h,w))
    for i in range(h):
        for j in range(w):
            if mat_proba[i,j]>p:
                img_out[i,j] = img[i,j]
            else:
                img_out[i,j] = ruido[i,j]
    return img_out

pim = io_image.read_image("data/images/pim.jpg")
# img_ruido, img_out = agrega_ruido_gaussiano(pim, 1)
# img_ruido_multipicativo = agregar_ruido_multiplicativo(pim, 2)
# plt.imshow(img_ruido_multipicativo,cmap='grey')
# plt.imshow(img_out,cmap='grey')
# plt.show()

# uwu = agregar_ruido_multiplicativo(pim,0.5)
# plt.imshow(uwu, cmap='grey')
# plt.show()

def promediarImagenes(img, n_imagenes, sigma):
    h, w = img.shape
    img_promedio = np.empty((h,w))
    for i in range(n_imagenes):
        imagen_ruido = agrega_ruido_gaussiano(img,sigma)
        img_promedio = img_promedio + imagen_ruido
    return img_promedio/n_imagenes