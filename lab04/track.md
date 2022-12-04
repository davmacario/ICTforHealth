# Laboratory 04

## Description

Goal: detect actividy given the measurements done by 15 sensors (accelerometer, gyroscope, magnetometer) - each of these placed on torso, left arm, right arm, left leg, right leg of subjects moving.

Measurements taken at 25 Hz and for each measure there are 3 values (x, y, z).

The number of subjects is 8, however only one will be analyzed, depending on the student ID.

Each activity is performed for 5 minutes by the subjects and measurements are gathered in files containing slices of 5 seconds (125 rows, 45 cols).

## ToDo

Develop a simple algorithm for classifying the data and correctly devise the activity.
The chosen algorithm is K-means (clustering applied to classification):

* Read first *M* slices for each of the 19 activities if the selected subject
* Apply K-means to find the 19 centroids
* For each row of the subsequent slices, detect the activity by finding the closest centroid

Can **use less than 15 sensors** - fewer is better.

Need to **decide a preprocessing strategy to maximize final accuracy** - knowing that K-means MUST be used.
The ultimate goal is to produce a ***simple strategy*** (light computational cost and quick).

Can use **at most 30 (/60) slices in the training phase** and can do less than 1 decision per timestamp (undersampling), e.g., 1 decision per second (every 25 samples).
