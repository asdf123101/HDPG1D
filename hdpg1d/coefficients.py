class coefficients:
    def __init__(self, diff, conv, flux, porder, nele):
        self.diffusion = diff
        self.covection = conv
        self.flux = flux
        self.porder = porder
        self.nele = nele

    @classmethod
    def from_input(cls):
        while True:
            try:
                print("Please provide the following coefficients.")
                diff = float(input("Diffusion coefficient: "))
                conv = float(input("Covection coefficient: "))
                flux = float(input("Flux: "))
                porder = int(input("Order of polynomials: "))
                nele = int(input("Number of elements: "))
            except ValueError:
                print("Sorry, wrong data type.")
                continue
            else:
                break
        return cls(diff, conv, flux, porder, nele)
