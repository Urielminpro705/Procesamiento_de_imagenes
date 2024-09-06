import utils.io_image as io
from utils.operaciones_img import promediarImagenes
import matplotlib.pyplot as plt 

img = io.read_image("data/images/pim.jpg")
pruebas = [
    io.image_bit_combination(img, [0,1,2,3,4,5,6,7]),
    io.image_bit_combination(img, [4,5,6,7]),
    io.image_bit_combination(img, [0,1,2,3])
]
io.planes_print(io.extract_bit_planes(img),["Plano 7","Plano 6","Plano 5","Plano 4","Plano 3","Plano 2","Plano 1","Plano 0"],2,4)
io.planes_print(pruebas,["Todos los planos","[4,5,6,7]","[0,1,2,3]"],1,3)