import os
import numpy as np


for dir in os.listdir('.'):
    if os.path.isdir(dir):
        mean_arr = []
        f1_arr = []
        for i in range(0, 5):
            data = np.loadtxt(dir + '/trial_' + str(i) + '/results/test_accuracy.txt')
            mean = np.mean(data[80:])
            mean_arr.append(mean)
            data = np.loadtxt(dir + '/trial_' + str(i) + '/results/test_f1.txt')
            f1_mean = np.mean(data[80:])
            f1_arr.append(f1_mean)
        print(dir)
        print(np.array(mean_arr))
        print(np.mean(np.array(mean_arr)))
        print("F1")
        print(np.array(f1_arr))
        print(np.mean(np.array(f1_arr)))


