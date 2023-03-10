import matplotlib.pyplot as plt

x = [1, 2, 3, 4, 5, 6]
y = [6, 5, 3, 5, 7, 8]

plt.xlabel('Class')
plt.ylabel('CGPA')

plt.scatter(x, y)
plt.show()