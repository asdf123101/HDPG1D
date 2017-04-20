menu = {}
menu['1'] = "Solve with HDG."
menu['2'] = "Solve with HDPG."
menu['3'] = "Exit."

options = menu.keys()
for entry in options:
    print(entry, menu[entry])
    
selection = input("Please Select:")
if selection == '1':
    print("test" )
elif selection == '2':
    print("test")
elif selection == '3':
    print("test")
else:
    print("Unknown Option Selected!")