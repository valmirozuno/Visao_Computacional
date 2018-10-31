import Image, numpy, math
I = Image.open('lion.jpg')
I = I.convert('L')
a = numpy.asarray(I)
A = numpy.fft.fft2(a)
A = numpy.fft.fftshift(A)

for u in range(I.size[0]):
  for v in range(I.size[1]):
    u1 = u - I.size[0]/2
    v1 = v - I.size[1]/2
    Duv = math.sqrt(u1*u1 + v1*v1)
    D0 = 200.0
    #ordem do filtro
    n = 2
    Huv = 1.0/((1+D0/(Duv+0.0001))**(2*n))
    A[v][u] = Huv*A[v][u]

Image.fromarray(abs(A)/800).show()
A = numpy.fft.fftshift(A)

output = Image.fromarray(numpy.fft.ifft2(A).astype(numpy.uint8))
output.show()