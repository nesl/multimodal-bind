
## Download instructions

Download the processed PAMAP dataset at this link: https://drive.google.com/file/d/1WaNM3fGJ8VEsCBe3weRXceDRGjokiph6/view?usp=sharing

Place it in the same directory as this cloned repository 

i.e: 

parent directory

            PAMAP-Code
            
            PAMAP_Dataset
            
The folder should be named as PAMAP_Dataset, and its direct children directories should be trainA, trainB, trainC, and test


## Data Format

Currently, the first number in the filename represents the action performed. The second number is simply an index to prevent duplicate files, and can be disregarded.

The each data sample has shape 1000 x 54 (100 Hz sampling rate for 10 seconds), and the 54 fields include other data + imu, gyro, mag. Note that this only wrist IMU data
