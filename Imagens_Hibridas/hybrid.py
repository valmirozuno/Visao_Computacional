# coding=utf-8
import cv2 as cv
import numpy as np
import math

#passa alta

img = cv.imread('cato.jpg', 0)
b = np.asarray(img)

filtros = np.zeros(b.shape)
b.flags.writeable = True
filtros.flags.writeable = True

# filtro da media tamanho M por M
N = 3
for i in range(N):
    for j in range(N):
        if j == 0:
                filtros[i][j] = -1
        elif j ==2:
                filtros[i][i] = 1

R = np.fft.fft2(filtros)
B = np.fft.fft2(b)

# no dominio da frequencia o produto corresponde a convolucao no dominio espacial
f_ishift = np.multiply(R, B)

#passa baixa

I = cv.imread('dogo.jpg', 0)
a = np.asarray(I)

filtro = np.zeros(a.shape)
a.flags.writeable = True
filtro.flags.writeable = True

# filtro da media tamanho M por M
M = 20
for i in range(M):
    for j in range(M):
        filtro[i][j] = 1.0 / (M * M)

S = np.fft.fft2(filtro)
A = np.fft.fft2(a)

# no dominio da frequencia o produto corresponde a convolucao no dominio espacial
res = np.multiply(S, A)

hybrid = res + f_ishift

cv.namedWindow('Res')
cv.imshow('Res', np.array(np.fft.ifft2(hybrid).astype(np.uint8)))

cv.imwrite("hybrid.jpg",np.array(np.fft.ifft2(hybrid).astype(np.uint8)))


cv.waitKey(0)
cv.destroyAllWindows()
