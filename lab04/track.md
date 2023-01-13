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

### Updates

* **2022-12-09**: Still, the best clustering separation is achieved by just considering magnetometer
* **2022-12-12**: the mapping of centroids is obtained correctly (and easily) when initializing the k-means centroids to the centroids obtained as the average elements of each activity (from prof. code)
* **2022-12-13**: created a function to perform extraction and get centroids of the data; it allows to easily create and manage training and test sets
* **2022-12-15**: found best combination of 12 sensors that yield the greatest accuracy
* **2022-12-16**: added denoiser program, based on fft
* **2022-12-19**: added PCA, tried it on more features (all but gyro)
* **2022-12-20**: added function for plotting confusion matrix; accuracy at 86.27% (PCA from 16 to 15)
* **2022-12-26**: started working on 'develop' branch - don't know what to do... nothing seems to work better
* **2022-12-27**: after finding top 10 sensors, the results are slightly within the bounds (85.02 acc. on test); the errors are mainly concentrated around: mov.elev, exer.step, play.bb. It turns out I completely messed up how the filter was applied and was now able to find the best filter to achieve a performance of 91.4% accuracy on the test set, using a bandstop filter with cutoffs 0.01 and 12 Hz. This yields the best result since it keeps the DC value (very different between the activities) and the high-frequency components which change a lot between the activities as well (e.g., running and standing).
* **2022-12-28**: tested performance on other patients.
* **2023-01-13**: changed confusion matrix type - now it displays normalized values ($P(\hat{c}=i | c = j)$)

### Possible improvements

A good idea could be that of **working in frequency domain** only... probably filtering will not be necessary, even if it won't be trivial to make the shift to frequency domain, since the resolution depends on the number of samples. Then zero-padding will be needed in order to increase the frequency resolution for the training set, which has less samples than the test set.

Another possible improvement could be that of taking as values of the signals the autocorrelation.

**Problem**: for both these approaches, it is not possible to determine *F-dimensional* centroids (being *F* the number of features), since it does not make sense to only take 1 value for each feature (for the centroids) if we are interested in comparing the overall shape.
This limits the approach to be time-domain only.
