import pywt
import cv2
import numpy as np

class Images:

    # Imagen original
    img = None
    # Imagen en 256x256
    imagen = None
    # Imagen con el Haar aplicado
    imagen2 = None
    imagenCH = None
    imagenCV = None
    imagenCD = None
    # Vector característica
    vectorC = []

    #Vector de vectores
    FVector = []

    def AbrirImagen(self, name):
        #carga imagen
        self.img = cv2.imread(name)
        cv2.imshow('Imagen',self.img)
        # print("matriz:",self.img)

    def ResizeImage(self, img):
        mat = cv2.imread(img)
        self.imagen = cv2.resize(mat, (256, 256))
        #cv2.imshow('Imagen 256x256',self.imagen) #Mostar img
        # print("matriz:",self.imagen)
        #print("matriz:",self.imagen.shape)
        return self.imagen

    #                             -------------------
    #                             |        |        |
    #                             | cA(LL) | cH(LH) |
    #                             |        |        |
    # (cA, (cH, cV, cD))  <--->   -------------------
    #                             |        |        |
    #                             | cV(HL) | cD(HH) |
    #                             |        |        |
    #                             -------------------

    def HaarWavelet(self,image):

        #convert image to float
        fimage = np.float32(image)
        # print("Fimage:",fimage)

        # compute coefficients
        # coeffs = pywt.dwt2(fimage, "haar", level=1)
        # cA, (cH, cV, cD) = coeffs
        A , (H, V, D) = pywt.dwt2(fimage,"haar")

        A = np.uint8(A)
        # cv2.imshow("haarA:",A)
        H = np.uint8(H)
        V = np.uint8(V)
        D = np.uint8(D)

        return A, (H,V,D)

    def WaveletRGB(self, image):
        # #convert to grayscale
        # imArray = cv2.cvtColor( img,cv2.COLOR_RGB2GRAY )

        #Usar un canal para cada uno:
        B,G,R = cv2.split(image)

        B, (bH, bV, bD) = self.HaarWavelet(B)
        G, (gH, gV, gD) = self.HaarWavelet(G)
        R, (rH, rV, rD) = self.HaarWavelet(R)

        #Juntamos los haar de los tres canales, para formar una unica imagen
        image2 = cv2.merge([B,G,R])
        #cv2.imshow('haar:',image2) #Mostar haar
        #print("imagen2:", image2.shape)
        H = cv2.merge([bH,gH,rH])
        V = cv2.merge([bV,gV,rV])
        D = cv2.merge([bD,gD,rD])

        #Retorna imagenes a colores
        return image2, (H, V, D)

    def Haar_level3(self,image):
        self.imagen2, (cH, cV, cD) = self.WaveletRGB(image)
        self.imagen2, (cH, cV, cD) = self.WaveletRGB(self.imagen2)
        self.imagen2, (cH, cV, cD) = self.WaveletRGB(self.imagen2)
        return self.imagen2, (cH, cV, cD)

    def VectorCaracteristico(self, nombre):
        resized = self.ResizeImage(nombre)
        imagen2, C = self.Haar_level3(resized)
        coeffs = []
        coeffs.extend((imagen2,C[0],C[1],C[2]))

        # print("Imagen con haar:",self.imagen2)
        # Para cada uno sacar por R,G,B sus max, min, prom, std
        self.vectorC=[]

        for imagen in coeffs:
            #Minimos B,G,R de imagen
            minB = np.array(imagen)[...,0].min()
            minG = np.array(imagen)[...,1].min()
            minR = np.array(imagen)[...,2].min()

            #Maximos B,G,R
            maxB = np.array(imagen)[...,0].max()
            maxG = np.array(imagen)[...,1].max()
            maxR = np.array(imagen)[...,2].max()

            #Promedios B,G,R
            prom, desv = cv2.meanStdDev(imagen)
            prom = prom[:3]
            print("promedios:",prom)

            #Desviacion Estandar B,G,R
            desv = desv[:3]
            print("desviaciones:",desv)

            self.vectorC.extend( (minB,minG,minR,maxB,maxG,maxR))
            self.vectorC.extend((prom[0][0],prom[1][0],prom[2][0]))
            self.vectorC.extend((desv[0][0],desv[1][0],desv[2][0]))

        print("vector carac:",self.vectorC)

    def FeatureVectors(self,archivos):
        #f = open(archivo,"r")
        #print("archivo:",archivo
        #for line in f:
        self.FVector=[]
        for line in archivos:
            self.VectorCaracteristico(line)
            self.FVector.append(self.vectorC)
        print(self.FVector)
        return self.FVector

# vect = Images()
# vect.AbrirImagen("gato1.jpg")
# vect.ResizeImage("gato1.jpg")
# vect.HaarWavelet(vect.imagen)
# vect.Haar_level3(vect.imagen)
# vect.VectorCaracteristico("gato1.jpg")
# VectorGatos = vect.FeatureVectors("gatos.txt")
# gatos.txt (Para entrenar -> Son 10)
# perros.txt (Para entrenar -> Son 10)
# cat.txt (Para testear -> Son 5)
# dog.txt (Para testear -> Son 5)
