import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate
import io_image as io


def histogram(img : np.array, bins : int, normalizar : bool = False):
    h,w = img.shape
    hist = np.zeros((bins+1))
    for i in range(h):
        for j in range(w):
            valor = img[i,j]
            hist[valor] += 1
    
    return hist/(h*w) if normalizar else hist

def histogram_equalization(img, bins):
    frec = histogram(img, bins, False)
    frec_normalizada = histogram(img, bins, True)
    probabilidad_acumulada = frec_normalizada.cumsum()
    probabilidad_por_luminosidad = probabilidad_acumulada * bins
    nueva_intensidad = np.floor(probabilidad_por_luminosidad[:-1])
    nueva_intensidad = np.append(nueva_intensidad, probabilidad_por_luminosidad[-1])
    lut = np.array([
        list(range(0,bins+1)),
        frec,
        frec_normalizada,
        probabilidad_acumulada,
        probabilidad_por_luminosidad,
        nueva_intensidad
        ])
    h,w = img.shape
    img_equializada = np.zeros((h,w))
    for f in range(h):
        for c in range(w):
            img_equializada[f,c] = lut[5][img[f,c]]
    return img_equializada, lut.transpose() 

def print_lut(lut):
    titulos = ["Intensidades","Frecuencia","F.Normalizada","3","4","5"]
    print(tabulate(lut, headers = titulos, tablefmt="grid"))

def print_histogram(img : np.array, hist : np.array,titulos : list = ["Imagen","Histograma"]):
    figura, eje = plt.subplots(1,2)
    eje[0].imshow(img, cmap="gray")
    eje[0].set_title(titulos[0])
    eje[0].axis("off")

    eje[1].plot(hist)
    eje[1].set_title(titulos[1])

    plt.tight_layout()
    plt.show()

# Negativo de una imagen
def negativo_img(img, l):
    return (l-1)-img

# def log_img(img, c):
#     h,w = img.shape
#     img_trans = np.zeros((h,w))
#     for f in range(h):
#         for j in range(w):
#             img_trans[f,j] = c*np.log((1 + img[f,j]))
#     return np.floor(img_trans).astype(np.uint8)

def log_img(img, c):
    img_trans = c * np.log1p(img)
    return np.floor(img_trans).astype(np.uint8)

def gamma_img(img, c, gamma):
    img = img.astype(np.float32) / 255
    img_gamma = c * (img ** gamma)
    img_rescaled = np.clip(img_gamma * 255, 0, 255)
    
    return img_rescaled.astype(np.uint8)

L = 2**8
img = io.read_image("data/images/pim.jpg")
out_img = negativo_img(img, L)
histograma = histogram(out_img,255)
print_histogram(out_img, histograma)

img_trans = log_img(img,10)
histograma = histogram(img_trans,255)
print_histogram(img_trans,histograma)

img_trans = gamma_img(img,10,3)
histograma = histogram(img_trans,255)
print_histogram(img_trans,histograma)