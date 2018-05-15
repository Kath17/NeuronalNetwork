import numpy as np


class RN:

    NNI=None
    NNH=None
    NNO=None

    Wh=None
    Wo=None


    NetO = None
    NetH = None

    InputI = None
    InputO = None

    errorMin = 0.01
    alpha = 0.5

    def __init__(self,Si,Sh,So):
        self.NNI = Si
        self.NNH = Sh
        self.NNO = So
        self.Wh = np.random.rand( self.NNI+1,self.NNH )
        self.Wo = np.random.rand( self.NNH+1, self.NNO )


    #Función de Activación
    def activate(self,v):
        return 1/(1+np.exp(-v))

    def forward(self,I,Sd):
        print("Forward")
        self.InputI = np.asmatrix(np.append(1,np.array(I)))
        print("NetH: ", np.shape(self.InputI), ".", np.shape(self.Wh))
        self.NetH = self.activate(np.matmul( self.InputI , self.Wh ))
        self.InputO  = np.asmatrix(np.append(1,self.NetH ))
        print("NetO: ", np.shape(self.InputO), ".", np.shape(self.Wo))
        self.NetO = np.asmatrix(self.activate( np.matmul( self.InputO, self.Wo)))
        return np.sum((np.square( Sd - self.NetO))/2)

    def backward(self,Sd):
        print("Backward")
        #Regla de las deltas de la capa Output
        print("DeltaO: (", np.shape(self.NetO),"-",np.shape(Sd),") X",np.shape(self.NetO),"X",np.shape((1 - self.NetO)) )
        DeltaO  = np.multiply((self.NetO - Sd),(self.NetO) ,(1 - self.NetO))
        print("varO: ",np.shape(self.InputO.T),"x",np.shape(DeltaO))
        varO    = np.matmul(self.InputO.T,DeltaO)


        sig_prime   = np.multiply((self.NetH),(1 - self.NetH))
        print("DeltaH: sum(",np.shape(np.asmatrix(np.append(1,DeltaO))) ,".", np.shape(self.Wh.T) ,")x",np.shape(sig_prime)  )
        DeltaH      = np.sum(np.matrix.dot(np.asmatrix(np.append(1,DeltaO)),self.Wh.T)) * sig_prime
        print("varH: ",np.shape(self.InputI.T),".",np.shape(DeltaH))
        varH        = np.matmul(self.InputI.T,DeltaH)


        self.Wo = self.Wo - self.alpha*varO
        print(np.shape(self.Wo),"-",np.shape(varO))
        self.Wh = self.Wh - self.alpha*varH
        print(np.shape(self.Wh),"-",np.shape(varH))



    def learn(self,fileTraInputIng):
        print("Learning")




a=RN(4,2,1)
print ("Error:",a.forward( [2,3,4,5],np.array([1]) ) )
a.backward(np.array([1]))
