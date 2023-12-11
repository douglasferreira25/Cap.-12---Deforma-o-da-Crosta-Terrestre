import numpy as np
import matplotlib.pyplot as plt
import matplotlib.collections


file = open(r"C:\Users\ngeot\OneDrive\Área de Trabalho\Database - Douglas - CAP12\Downloads\Websites\Dogbone_Tension.input")
nod = np.loadtxt(r'C:\Users\ngeot\OneDrive\Área de Trabalho\Database - Douglas - CAP12\Downloads\Websites\Dogbone_Tension.input', skiprows = 1, max_rows=1428, delimiter = ',')
elm = np.loadtxt(r'C:\Users\ngeot\OneDrive\Área de Trabalho\Database - Douglas - CAP12\Downloads\Websites\Dogbone_Tension.input', skiprows = 1430, max_rows=1298, delimiter = ',')
bond1 = np.loadtxt(r'C:\Users\ngeot\OneDrive\Área de Trabalho\Database - Douglas - CAP12\Downloads\Websites\Dogbone_Tension.input', skiprows = 2730, max_rows=12, delimiter = ',')
bond2 = np.loadtxt(r'C:\Users\ngeot\OneDrive\Área de Trabalho\Database - Douglas - CAP12\Downloads\Websites\Dogbone_Tension.input', skiprows = 2743, max_rows=12, delimiter = ',')



n1 = np.array(list(map(np.int_, elm[:,1])))-1
n2 = np.array(list(map(np.int_, elm[:,2])))-1
n3 = np.array(list(map(np.int_, elm[:,3])))-1
n4 = np.array(list(map(np.int_, elm[:,4])))-1

elem=[]
elem.append([n1,n2,n3,n4])

xx = np.array(list(map(np.float_, nod[:,1])))
yy = np.array(list(map(np.float_, nod[:,2])))


elem = []
elem.append([n1,n2,n3,n4])

for element in elem:
        x = [xx[element[i]] for i in range(len(element))]
        y = [yy[element[i]] for i in range(len(element))]
        
        plt.figure(figsize=(12,6))
        plt.plot(x, y)