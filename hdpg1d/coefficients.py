class coefficients:
    def __init__(self, diff, conv, reaction, pOrder, numEle, tauPlus, tauMinus):
        if diff == 0:
            # set the diffusion constant to a small number
            # to avoid division by zero error
            diff = 1e-16
        self.DIFFUSION = diff
        self.CONVECTION = conv
        self.REACTION = reaction
        self.pOrder = pOrder
        self.numEle = numEle
        self.TAUPLUS = tauPlus
        self.TAUMINUS = tauMinus

    @classmethod
    def fromInput(cls):
        while True:
            try:
                print("Please provide the following coefficients.")
                diff = float(input("Diffusion constant (float): "))
                conv = float(input("Covection constant (float): "))
                reaction = float(input("Reaction constant (float): "))
                pOrder = int(input("Order of polynomials (int): "))
                numEle = int(input("Number of elements (int): "))
                tauPlus = float(input("Stablization parameter plus (float): "))
                tauMinus = float(
                    input("Stablization parameter minus (float): "))
            except ValueError:
                print("Sorry, wrong data type.")
                continue
            else:
                break
        return cls(diff, conv, reaction, pOrder, numEle, tauPlus, tauMinus)
