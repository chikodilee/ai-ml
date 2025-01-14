# ECE657A - Winter 2024 - Data Cleaning and Dimensionality Reduction
### Contributors

- Prahar Ijner
- Chikodili Odinakachukwu
- Billy Qiu

The code files have been tested to work on Python version 3.11.6
The packages required to run the code files can be found in `requirements.txt`

Objective:
To study how to apply ML methods on three datasets. The emphasis is on analysis and presentation of results not on code implemented or used. 

Dataset A :
This is a time-series dataset which is collected from a set of motion sensors for wearable
activity recognition. The data is given in time order, with 19,000 samples and 81
features. Some missing values are denoted by Not Available (NA) and also some outliers
are present. (note: The negative values are not outliers) This data is used to illustrate the
data cleaning and preprocessing techniques. (File: DataA.csv)

Dataset B :
Handwritten digits of 0, 1, 2, 3, and 4 (5 classes). This dataset contains 2066 samples
with 784 features corresponding to a 28 x 28 gray-scale (0-255) image of the digit,
arranged in column-wise. This data is used to illustrate the difference between feature
extraction on methods. (File: DataB.csv)