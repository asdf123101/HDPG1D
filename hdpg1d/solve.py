from .coefficients import coefficients
from .adaptation import hdpg1d
from .postprocess import utils


def queryYesNo(question, default="yes"):
    valid = {"yes": True, "y": True, "ye": True,
             "no": False, "n": False}
    if default is None:
        prompt = " [y/n] "
    elif default == "yes":
        prompt = " [Y/n] "
    elif default == "no":
        prompt = " [y/N] "
    else:
        raise ValueError("invalid default answer: '%s'" % default)

    while True:
        print(question + prompt)
        choice = input().lower()
        if default is not None and choice == '':
            return valid[default]
        elif choice in valid:
            return valid[choice]
        else:
            print("Please respond with 'yes' or 'no' "
                  "(or 'y' or 'n').\n")


def getCoefficients():
    question = 'Do you want to use the default parameters?'
    isDefault = queryYesNo(question, "yes")
    if (isDefault):
        Coeff = coefficients(1e-6, 0, 1, 2, 2, 1, 1)
    else:
        Coeff = coefficients.fromInput()
    return Coeff


def menu():
    menu = {}
    menu['1.'] = "Solve with HDG."
    menu['2.'] = "Solve with HDPG."
    menu['3.'] = "Exit."

    for key, value in sorted(menu.items()):
        print(key, value)


def hdgSolve():
    hdgCoeff = getCoefficients()
    print("Solving...")
    hdgSolution = hdpg1d(hdgCoeff)
    # solve the problem adaptively and plot convergence history
    hdgSolution.adaptive()
    print("Problem solved. Please check the convergence plot.")
    utils(hdgSolution).convHistory()


def runInteractive():
    menu()
    selection = input("Please Select: ")
    while True:
        if selection == '1':
            hdgSolve()
            break
        elif selection == '2':
            print("In development...")
        elif selection == '3':
            print("Bye.")
            break
        else:
            print("Unknown Option Selected!")
            continue
