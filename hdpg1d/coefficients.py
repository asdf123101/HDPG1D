class coefficients:
    def __init__(self, diff, conv, flux, pOrder, numEle, tauPlus, tauMinus):
        self.diffusion = diff
        self.convection = conv
        self.flux = flux
        self.pOrder = pOrder
        self.numEle = numEle
        self.tauPlus = tauPlus
        self.tauMinus = tauMinus

    @classmethod
    def fromInput(cls):
        while True:
            try:
                print("Please provide the following coefficients.")
                diff = float(input("Diffusion coefficient: "))
                conv = float(input("Covection coefficient: "))
                flux = float(input("Flux: "))
                pOrder = int(input("Order of polynomials: "))
                numEle = int(input("Number of elements: "))
                tauPlus = float(input("Stablization parameter plus: "))
                tauMinus = float(input("Stablization parameter minus: "))
            except ValueError:
                print("Sorry, wrong data type.")
                continue
            else:
                break
        return cls(diff, conv, flux, pOrder, numEle, tauPlus, tauMinus)
