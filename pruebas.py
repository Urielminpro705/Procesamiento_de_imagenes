import utils.io_image as io
from utils.operaciones_img import promediarImagenes, agrega_ruido_gaussiano
import matplotlib.pyplot as plt 
import numpy as np
from utils import histograma as hi
import time
from utils.filters import conv2d, filtro_promedio, gaussian_kernel
from scipy.signal import convolve2d, correlate2d

img = io.read_image("data/images/pim.jpg")

# pruebas = [
#     io.image_bit_combination(img, [0,1,2,3,4,5,6,7]),
#     io.image_bit_combination(img, [4,5,6,7]),
#     io.image_bit_combination(img, [0,1,2,3])
# ]
# io.planes_print(io.extract_bit_planes(img),["Plano 7","Plano 6","Plano 5","Plano 4","Plano 3","Plano 2","Plano 1","Plano 0"],2,4)
# io.planes_print(pruebas,["Todos los planos","[4,5,6,7]","[0,1,2,3]"],1,3)

# sigma = 256
# img1 = promediarImagenes(img, 1,sigma)
# img2 = promediarImagenes(img, 2,sigma)
# img4 = promediarImagenes(img, 4,sigma)
# img8 = promediarImagenes(img, 8,sigma)
# img16 = promediarImagenes(img, 16,sigma)
# img32 = promediarImagenes(img, 32,sigma)
# img64 = promediarImagenes(img, 64,sigma)
# img128 = promediarImagenes(img, 128,sigma)
# img256 = promediarImagenes(img, 256,sigma)

# imgs = np.array([img1,img2,img4,img8,img16,img32,img64,img128,img256])
# io.planes_print(imgs,["1","2","4","8","16","32","64","128","256"],2,5)


# q_img = io.quantize(img,3)
# # io.print_img(q_img,"Cuantizado a 3 bits")
# h,w = q_img.shape
# pixeles = h*w
# histograma = hi.histogram(q_img,7,False)
# print(pixeles)
# print(sum(histograma))
# print(histograma)


# img = io.quantize(img,3)
# img_equalizada, lut = hi.histogram_equalization(img, 7)
# histograma1 = hi.histogram(img, 7)
# histograma2 = hi.histogram(img_equalizada, 7)
# hi.print_histogram_comparation(img, histograma1, img_equalizada, histograma2, ["Original","Histograma"], ["Ecualizada","Histograma"])
# hi.print_lut(lut)



# hi.print_histogram(img, hi.histogram(img, 255, False))
# img_out = hi.fun_trozo1(img, 200)
# io.planes_print([img, img_out],["Normal","Transformada"],1,2)

# hi.print_histogram(img, hi.histogram(img, 255, False))
# img_out = hi.fun_trozo2(img, 185,200)
# io.planes_print([img, img_out],["Normal","Transformada"],1,2)


# x = np.arange(0,256)
# img = io.read_image("data/images/pim.jpg")
# tiempos = []
# imagenes = []

# start = time.time()
# y = hi.T(x,185,200,100)
# # img_out = hi.fun_trozo3(img, 185,200, 100)
# img_out = hi.fun_trans_LUT(img, y)
# end = time.time()
# elapsed = end - start
# tiempos.append(elapsed)
# imagenes.append(img_out)

# img = io.read_image("data/images/flor1.jpg")
# start = time.time()
# y = hi.T(x,185,200,100)
# # img_out = hi.fun_trozo3(img, 185,200, 100)
# img_out = hi.fun_trans_LUT(img, y)
# end = time.time()
# elapsed = end - start
# tiempos.append(elapsed)
# imagenes.append(img_out)

# img = io.read_image("data/images/moon.jpg")
# start = time.time()
# y = hi.T(x,185,200,100)
# # img_out = hi.fun_trozo3(img, 185,200, 100)
# img_out = hi.fun_trans_LUT(img, y)
# end = time.time()
# elapsed = end - start
# tiempos.append(elapsed)
# imagenes.append(img_out)

# img = io.read_image("data/images/pim.jpg")
# start = time.time()
# y = hi.T(x,185,200,100)
# img_out = hi.fun_trozo3(img, 185,200, 100)
# end = time.time()
# elapsed = end - start
# tiempos.append(elapsed)
# imagenes.append(img_out)

# img = io.read_image("data/images/flor1.jpg")
# start = time.time()
# y = hi.T(x,185,200,100)
# img_out = hi.fun_trozo3(img, 185,200, 100)
# end = time.time()
# elapsed = end - start
# tiempos.append(elapsed)
# imagenes.append(img_out)

# img = io.read_image("data/images/moon.jpg")
# start = time.time()
# y = hi.T(x,185,200,100)
# img_out = hi.fun_trozo3(img, 185,200, 100)
# end = time.time()
# elapsed = end - start
# tiempos.append(elapsed)
# imagenes.append(img_out)

# print(f"Toma {elapsed:.6f} segundos")
# plt.plot(x,y)
# plt.show()

# # hi.print_histogram(img, hi.histogram(img, 255, False))
# io.planes_print(imagenes,tiempos,2,3)

kernel = np.array([
    [-1,-2,1],
    [0,0,0],
    [1,2,1]])

kernel = filtro_promedio(9)
kernel = gaussian_kernel(20, 3)
img = agrega_ruido_gaussiano(img, 20)
img_conv = conv2d(img, kernel)
img_conv2 = convolve2d(img, kernel, mode='full', boundary='fill', fillvalue=0)
io.planes_print([img,img_conv,img_conv2],["Original","Convolucion mio","Convolucion"],1,3)