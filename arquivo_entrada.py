import fileinput

for line in fileinput.input(files=('Dogbone_Tension.input')):
    # Process the input line
    print(line, end='')
