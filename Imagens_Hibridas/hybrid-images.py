import numpy
from numpy.fft import fft2, ifft2, fftshift, ifftshift
from scipy import misc
from scipy import ndimage
import math

def scaleSpectrum(A):
   return numpy.real(numpy.log10(numpy.absolute(A) + numpy.ones(A.shape)))


#Implementa o filtro gaussiano, que será usado para criar os filtros passa alta e passa baixa
def makeGaussianFilter(numRows, numCols, sigma, highPass=True):
   centerI = int(numRows/2) + 1 if numRows % 2 == 1 else int(numRows/2)
   centerJ = int(numCols/2) + 1 if numCols % 2 == 1 else int(numCols/2)

   def gaussian(i,j):
      coefficient = math.exp(-1.0 * ((i - centerI)**2 + (j - centerJ)**2) / (2 * sigma**2))
      return 1 - coefficient if highPass else coefficient

   return numpy.array([[gaussian(i,j) for j in range(numCols)] for i in range(numRows)])


# implementa o filtro da transformada de Fourier
def filterDFT(imageMatrix, filterMatrix):
   shiftedDFT = fftshift(fft2(imageMatrix))
   filteredDFT = shiftedDFT * filterMatrix
   return ifft2(ifftshift(filteredDFT))

#implementa o filtro passa baixa
#Entradas:
#ImageMatrix -> Matriz que representa a imagem
#sigma -> Limiar para filtrar
def lowPass(imageMatrix, sigma):
   n,m = imageMatrix.shape
   return filterDFT(imageMatrix, makeGaussianFilter(n, m, sigma, highPass=False))

#implementa o filro passa alta
#Entradas:
#ImageMatrix -> Matriz que representa a imagem
#sigma -> Limiar para filtrar
def highPass(imageMatrix, sigma):
   n,m = imageMatrix.shape
   return filterDFT(imageMatrix, makeGaussianFilter(n, m, sigma, highPass=True))

#Função que cria a imagem híbrida
#Entradas:
#highFreqImg -> representa a imagem com as altas frequências
#sigmaHigh -> Limiar para filtrar a imagem com altas frequências
#lowFreqImg -> representa a imagem com as baixas frequências
#sigmaLow -> Limiar para filtrar a imagem com baixas frequências
def hybridImage(highFreqImg, lowFreqImg, sigmaHigh, sigmaLow):
   highPassed = highPass(highFreqImg, sigmaHigh)
   lowPassed = lowPass(lowFreqImg, sigmaLow)

   return highPassed + lowPassed

#Função para testar os filtros e as imagens híbrigas
def testFiltering():
   marilyn = ndimage.imread("marilyn.png", flatten=True)

   highPassedMarilyn = highPass(marilyn, 20)
   lowPassedMarilyn = lowPass(marilyn, 20)

   misc.imsave("low-passed-marilyn.png", numpy.real(lowPassedMarilyn))
   misc.imsave("high-passed-marilyn.png", numpy.real(highPassedMarilyn))

   #a soma das imagens filtradas deve ser igual a imagem original
   misc.imsave("sum-of-marilyns.png", numpy.real((highPassedMarilyn + lowPassedMarilyn)/2.0))

if __name__ == "__main__":
   einstein = ndimage.imread("einstein.png", flatten=True)
   marilyn = ndimage.imread("marilyn.png", flatten=True)

   hybrid = hybridImage(einstein, marilyn, 25, 10)
misc.imsave("marilyn-einstein.png", numpy.real(hybrid))
