import os
import numpy as np

for dir in os.listdir('./'):
    if os.path.isdir(dir) and 'save' in dir:
        file_means_str = []
        file_means = []
        root = './' + dir + '/results/test_accuracy_'
        for i in range(41, 46):
            file = root + str(i) + '.txt'
            data = np.loadtxt(file)
            mean = np.mean(data[80:190])
            file_means.append(mean)
            file_means_str.append(str(mean) + '\n')
        with open('./' + dir + '/means.txt', 'w') as handle:
            handle.writelines(file_means_str)
            handle.write('\n')
            handle.write(str(np.mean(np.array(file_means))))


