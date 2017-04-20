class coefficients:
    def __init__(self, diff, conv, flux):
        self.diffusion = diff
        self.covection = conv
        self.flux = flux

    @classmethod
    def from_input(cls):
        while True:
            try:
                diff = float(input("Diffusion coefficient "))
                conv = float(input("Covection coefficient: "))
                flux = float(input("Flux: "))
            except ValueError:
                print("Sorry, wrong data type.")
                continue
            else:
                break
        return cls(diff, conv, flux)
