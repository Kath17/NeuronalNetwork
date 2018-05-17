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
    alpha = 0.5

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
        return 1/(1+np.exp(-v))

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
        print(np.shape(self.Wo),"-",np.shape(varO))
        self.Wh = self.Wh - self.alpha*varH
        print(np.shape(self.Wh),"-",np.shape(varH))

    def entrenar(self,x,y,epocas):
        pos = 0
        for i in range(epocas):
            X = np.array(x[pos],ndmin=2)
            Y = np.array(y[pos],ndmin=2)
            output = self.forward(X)
            print(str(self.error(output, Y)))
            self.backward(X,Y,output)
            pos = (pos+1)%len(x)

    def entrenar2(self,x,y):
        pos=0
        X = np.array(x[pos],ndmin=2)
        Y = np.array(y[pos],ndmin=2)
        output = self.forward(X)
        error = self.error(output, Y)
        print(error)
        while( error > self.errorMin):
            self.backward(X,Y,output)
            X = np.array(x[pos],ndmin=2)
            Y = np.array(y[pos],ndmin=2)
            output = self.forward(X)
            error = self.error(output, Y)
            print(error)
            pos = (pos+1)%len(x)


def cargarData(file):
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

    return  np.array(xtemp,ndmin=2,dtype=float), np.array(ytemp,ndmin=2,dtype=float)

def main():
    a=RN(13,8,3)
    x,y = cargarData("wine.txt")
    a.entrenar(x,y,1000)

if __name__ == '__main__':
    main()
