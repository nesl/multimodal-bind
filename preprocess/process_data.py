import os
import numpy as np

valid_actions = [1, 3, 4, 12, 13, 16, 17]
data_arr = []
labels = []
subjects = []
counters = []
data_duration = 1000
root = "./"

# all files/directories in the PAMAP Dataset folder
for subject_dir in sorted(os.listdir(root)):
    # Pick only the directories and those with 'subject' in the name, e.g. subject101, 102...
    if os.path.isdir(root + subject_dir) and "subject" in subject_dir:

        # Access the action npy file with 1000 frames of the action (all 54 values included)
        for data_path in sorted(os.listdir(root + subject_dir + "/")):
            # Naming convention is action_#.npy
            action = int(data_path.split("_")[0])
            repeat = int(data_path.split("_")[1].split(".")[0])

            if repeat == 0:
                c = 0

            # Only get valid actions that all subjects have (not all subjects perform the actions)
            if action in valid_actions:
                raw_data = np.load(root + subject_dir + "/" + data_path)  # Load the appropriate file
                while len(raw_data) >= data_duration:
                    data_arr.append(raw_data[:data_duration])  # Split into chunks of data_duration
                    raw_data = raw_data[data_duration:]
                    labels.append(action)  # append the same label since same data
                    subjects.append(subject_dir)
                    counters.append(c)
                    c += 1

# X samples of 1000 x 54
data_arr = np.array(data_arr)
data_arr[np.isnan(data_arr)] = 0  # Data has missing values, we just set them to be equal to zero
labels = np.array(labels)
subjects = np.array(subjects)

print(len(data_arr), len(labels), len(subjects), len(counters))

data_dir = "./processed_data"
if not os.path.exists(data_dir):
    os.makedirs(data_dir)


repeat = set()
for i in range(len(data_arr)):
    label = labels[i]
    subject = subjects[i]
    data = data_arr[i]
    # np.save('./trainA/' + str(training_labelA[) + '_' + str(counter) + '.npy', training_dataA[i])
    np.save(f"processed_data/{subject}_{label}_{counters[i]}.npy", data)

    file_name = f"processed_data/{subject}_{label}_{counters[i]}.npy"

    if file_name in repeat:
        print(file_name)
    
    repeat.add(file_name)


print(len(os.listdir(data_dir)))
