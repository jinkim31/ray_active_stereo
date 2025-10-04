import numpy as np
import matplotlib.pyplot as plt

means = []
for i in range(1, 6):
    points = np.loadtxt(f'../clouds/{i}.xyz')   # shape (N, 3)

    """
    x, y, z = points[::10,0], points[::10,1], points[::10,2]
    plt.figure(figsize=(8,6))
    sc = plt.scatter(x, y, c=z, cmap='jet', s=5, vmin=-3, vmax=3)  # s=point siz
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("XY projection (Z as color)")
    plt.colorbar(sc, label="Z value")
    plt.axis("equal")   # keep aspect ratio
    plt.show()
    """

    print("Mean: ", np.mean(points, axis=0)[2])
    means.append(np.mean(points, axis=0)[2])
    #print("Stddev: ", np.std(points, axis=0)[2])
    #print("Min: ", np.min(points, axis=0)[2])
    #print("Max: ", np.max(points, axis=0)[2])
print(np.mean(means))