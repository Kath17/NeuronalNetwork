import numpy as np


class RN:

    NNI=None
    NNH=None
    NNO=None

    Wh=None
    Wo=None

    Bh=None
    Bo=None

    NetH = None

    errorMin = 0.01
    alpha = 0.1

    def __init__(self,Si,Sh,So):
        self.NNI = Si
        self.NNH = Sh
        self.NNO = So
        #Pesos
        self.Wh = np.array(np.random.rand( self.NNI,self.NNH ),ndmin=2,dtype=float)
        self.Wo = np.array(np.random.rand( self.NNH,self.NNO ),ndmin=2,dtype=float)
        #Bias
        self.Bh = np.array(np.random.rand(self.NNH),ndmin=2,dtype=float)
        self.Bo = np.array(np.random.rand(self.NNO),ndmin=2,dtype=float)

    #Función de Activación
    def sigmoidea(self,v):
        return (1.0)/((1.0)+np.exp(-v))

    def sigmoideaDeriv(self,v):
        return v * (1 - v)

    def error(self,So,Sd):
        return np.sum((np.square( Sd - So))/2)

    def forward(self,I):
        self.NetH  = (self.sigmoidea(np.dot(I , self.Wh ) + self.Bh))
        NetO       = (self.sigmoidea(np.dot( self.NetH, self.Wo) + self.Bo))
        return NetO

    def backward(self,I,Sd,So):
        DeltaO      = (So - Sd) * self.sigmoideaDeriv(So)
        DeltaH      = np.dot(DeltaO,self.Wo.T) * self.sigmoideaDeriv(self.NetH)

        varH        = np.dot(I.T,DeltaH)
        varO        = np.dot(self.NetH.T,DeltaO)

        self.Wo = self.Wo - self.alpha*varO
        self.Wh = self.Wh - self.alpha*varH

    def entrenar(self,x,y,epocas):
        pos = 0
        for i in range(epocas):
            X = np.array(x[pos],ndmin=2)
            Y = np.array(y[pos],ndmin=2)
            output = self.forward(X)
            #print(str(self.error(output, Y)))
            self.backward(X,Y,output)
            pos = (pos+1)%(len(x))

    def predecir(self,x,y):
        X = np.array(x,ndmin=2)
        print("PREDECIR: ", str(X))
        Y = np.array(y,ndmin=2)
        output = self.forward(X)
        error = self.error(output, Y)
        print("Se predijo:", str(output))
        print("Real:", str(Y))
        print("El error es:", str(error))
        bueno = False
        if(  (output.argmax()) == (Y.argmax())  ):
            bueno = True

        return bueno

def cargarData(file,mezclar):
    xtemp=[]
    ytemp=[]
    data = open(file,'r')
    lines = data.readlines()
    for i in lines:
        xtemp+= [i.rstrip('\n').split(',')[1:]]
        if( int(i[0]) == 1 ):
            ytemp+= [[0,0,1]]
        elif(int(i[0]) == 2):
            ytemp+= [[0,1,0]]
        else:
            ytemp+=[[1,0,0]]

    x,y = np.array(xtemp), np.array(ytemp)
    perm = np.random.permutation(x.shape[0]) #Permutar los datos
    if(mezclar==1):
        x = x[perm]
        y = y[perm]
    return  np.array(x,ndmin=2,dtype=float), np.array(y,ndmin=2,dtype=float)

def main():
    #Creación de RN.
    #Input,Nodos Capa Hidden, Output
    a=RN(13,10,3)
    #Cargar datos para entrenar
    x,y = cargarData("wine2.txt",1)
    x = x/np.amax(x,axis=0) #Normalizar
    a.entrenar(x,y,30000) #30000 iteraciones

    #Predecir nuevos datos
    x_pred,y_pred = cargarData("wine3.txt",1)
    x_pred = x_pred/np.amax(x_pred,axis=0)
    buenos = 0
    for i in range(len(x_pred)):
         if( a.predecir(x_pred[i],y_pred[i]) == True):
            print("Correcto")
            buenos+=1

    prom = buenos/len(x_pred)
    print("Promedio de buenos", prom*100)


#59(1) , 70(2), 47(3)  -> Original (wine.txt)
#35(1) , 42(2), 28(3)  -> Entrenar (wine2.txt)
#24(1) , 28(2), 19(3)  -> Predecir (wine3.txt)

if __name__ == '__main__':
    main()
