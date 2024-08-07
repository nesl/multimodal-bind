import os
import numpy as np

for dir in os.listdir('./'):
    if os.path.isdir(dir) and 'save' in dir:
        file_means_str = []
        file_means_str_f1 = []
        file_means = []
        file_means_f1 = []
        
        for i in range(41, 46):
            file = './' + dir + '/results_' + str(i) + '/test_accuracy.txt'
            data = np.loadtxt(file)
            mean = np.mean(data[80:190])
            file_means.append(mean)
            file_means_str.append(str(mean) + '\n')

            file = './' + dir + '/results_' + str(i) + '/test_f1.txt'
            data = np.loadtxt(file)
            mean = np.mean(data[80:190])
            file_means_f1.append(mean)
            file_means_str_f1.append(str(mean) + '\n')
        with open('./' + dir + '/acc_means.txt', 'w') as handle:
            handle.writelines(file_means_str)
            handle.write('\n')
            handle.write(str(np.mean(np.array(file_means))))
        with open('./' + dir + '/f1_means.txt', 'w') as handle:
            handle.writelines(file_means_str_f1)
            handle.write('\n')
            handle.write(str(np.mean(np.array(file_means_f1))))


