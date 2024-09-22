import numpy as np
import matplotlib.pyplot as plt
import io_image as io


def histogram(img : np.array, bins : int, normalizar : bool):
    h,w = img.shape
    hist = np.zeros((bins+1))
    for i in range(h):
        for j in range(w):
            valor = img[i,j]
            hist[valor] += 1
    
    return hist/(h*w) if normalizar else hist

def histogram_equalization(img, bins):
    histograma = histogram(img, bins, False)
    histograma_normalizado = histogram(img, bins, True)
    probabilidad_acumulada = histograma_normalizado.cumsum()
    probabilidad_por_luminosidad = probabilidad_acumulada * bins
    nueva_intensidad = np.floor(probabilidad_por_luminosidad[:-1])
    nueva_intensidad = np.append(nueva_intensidad,(np.ceil(probabilidad_por_luminosidad[-1])))
    lut = [
        histograma,
        histograma_normalizado,
        probabilidad_acumulada,
        probabilidad_por_luminosidad,
        nueva_intensidad
        
        ]
    h,w = img.shape
    img_equializada = np.zeros((h,w))
    print(lut[4])
    for f in range(h):
        for c in range(w):
            img_equializada[f,c] = lut[4][img[f,c]]
    return img_equializada

# img = np.array([[94,83,80],
#                 [115,94,0],
#                 [80,0,115]])

img = io.quantize(io.read_image("data/images/moon.jpg"),3)
img_equalizada = histogram_equalization(img, 7)
io.planes_print([img,img_equalizada],["Normal","Equalizada"],1,2)
# hist = histogram(img, 255, True).cumsum()
# plt.plot(range(256), histogram(img,255, True), label='Linea')
# plt.title("Histograma")
# plt.show()