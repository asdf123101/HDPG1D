from .preprocess import setDefaultCoefficients
from .adaptation import hdpg1d
from .postprocess import utils


def menu():
    menu = {}
    menu['1.'] = "Solve with HDG."
    menu['2.'] = "Solve with HDPG."
    menu['3.'] = "Exit."

    for key, value in sorted(menu.items()):
        print(key, value)


def hdgSolve():
    hdgCoeff = setDefaultCoefficients()
    print("Solving...")
    hdgSolution = hdpg1d(hdgCoeff)
    # solve the problem adaptively and plot convergence history
    hdgSolution.adaptive()
    print("Problem solved. Please check the convergence plot.")
    utils(hdgSolution).convHistory()


def runInteractive():
    while True:
        menu()
        selection = input("Please Select: ")
        if selection == '1':
            hdgSolve()
            break
        elif selection == '2':
            print("In development...")
        elif selection == '3':
            print("Bye.")
            break
        else:
            print("Unknown Option Selected!\n")
            continue
