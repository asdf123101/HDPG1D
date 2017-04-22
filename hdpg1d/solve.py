from .coefficients import coefficients
from .discretization import HDPG1d
from .postprocess import convHistory
import sys


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
        sys.stdout.write(question + prompt)
        choice = input().lower()
        if default is not None and choice == '':
            return valid[default]
        elif choice in valid:
            return valid[choice]
        else:
            sys.stdout.write("Please respond with 'yes' or 'no' "
                             "(or 'y' or 'n').\n")


def getCoefficients():
    question = 'Do you want to use the default parameters?'
    isDefault = queryYesNo(question, "yes")
    if (isDefault):
        Coeff = coefficients(1, 1, 0, 2, 2)
    else:
        Coeff = coefficients.from_input()
    return Coeff


def run():
    menu = {}
    menu['1.'] = "Solve with HDG."
    menu['2.'] = "Solve with HDPG."
    menu['3.'] = "Exit."

    for key, value in sorted(menu.items()):
        print(key, value)

    selection = input("Please Select:")
    if selection == '1':
        hdgCoeff = getCoefficients()
        hdgSolution = HDPG1d(hdgCoeff.nele, hdgCoeff.porder)
        trueError, estError = hdgSolution.adaptive()
        convHistory(trueError, estError)
    elif selection == '2':
        print("In development...")
    elif selection == '3':
        print("Bye.")
    else:
        print("Unknown Option Selected!")
