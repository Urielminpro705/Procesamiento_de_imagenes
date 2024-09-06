from scipy import ndimage
from utils import io_image as io
import numpy as np
import matplotlib.pyplot as plt

A_identidad = np.array([
    [1,0,0],
    [0,1,0],
    [0,0,1]
])


A_reflexion_x = np.array([
    [1,0,0],
    [0,-1,0],
    [0,0,1]
])

A_rotacion_angulo_25 = np.array([
    [0.99120281186,-0.13235175009,0],
    [0.13235175009,0.99120281186,0],
    [0,0,1]
])

A_traslacion = np.array([
    [1,0,200],
    [0,1,200],
    [0,0,1]
])

A_escala = np.array([
    [0.5,0,0],
    [0,0.5,0],
    [0,0,1]
])

img = io.read_image("data/images/pim.jpg")
h ,w = img.shape

img2 = ndimage.affine_transform(img, A_traslacion)
img3 = ndimage.affine_transform(img2, A_escala)
img4 = ndimage.affine_transform(img3, A_rotacion_angulo_25)

b = A_traslacion @ A_escala @ A_rotacion_angulo_25 

img5 = ndimage.affine_transform(img, b)
io.planes_print([img,img4,img5],1,3)

list = np.array([img,img2,img3,img4])
io.planes_print(list, 1, 4)
# plt.imshow(img4, cmap='grey')
# plt.show()