import io_image as io
import numpy as np
import matplotlib.pyplot as plt


img = io.read_image('data/images/pepper.jpg')
lista = [0,1,2,3,4,5,6,7]
img_lol = io.image_bit_combination(img,lista)
plt.imshow(img_lol, cmap='grey')
plt.show