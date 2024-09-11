import utils.io_image as io
from utils.operaciones_img import promediarImagenes
import matplotlib.pyplot as plt 
import numpy as np

img = io.read_image("data/images/pim.jpg")
# pruebas = [
#     io.image_bit_combination(img, [0,1,2,3,4,5,6,7]),
#     io.image_bit_combination(img, [4,5,6,7]),
#     io.image_bit_combination(img, [0,1,2,3])
# ]
# io.planes_print(io.extract_bit_planes(img),["Plano 7","Plano 6","Plano 5","Plano 4","Plano 3","Plano 2","Plano 1","Plano 0"],2,4)
# io.planes_print(pruebas,["Todos los planos","[4,5,6,7]","[0,1,2,3]"],1,3)

sigma = 256
img1 = promediarImagenes(img, 1,sigma)
img2 = promediarImagenes(img, 2,sigma)
img4 = promediarImagenes(img, 4,sigma)
img8 = promediarImagenes(img, 8,sigma)
img16 = promediarImagenes(img, 16,sigma)
img32 = promediarImagenes(img, 32,sigma)
img64 = promediarImagenes(img, 64,sigma)
img128 = promediarImagenes(img, 128,sigma)
img256 = promediarImagenes(img, 256,sigma)

imgs = np.array([img1,img2,img4,img8,img16,img32,img64,img128,img256])
io.planes_print(imgs,["1","2","4","8","16","32","64","128","256"],2,5)