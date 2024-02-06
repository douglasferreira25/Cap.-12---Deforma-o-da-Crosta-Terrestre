import fileinput

for line in fileinput.input(files=('Dogbone_Tension.input')):
    # Process the input line
    print(line, end='')

#-------------------------------------------------------------------------

with open('FEM_Solver_Completo.py', 'r') as file:
    content = file.read()
    # Process the content of the input file
    print(content)
