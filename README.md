# cifar10-ML
Image classification on CIFAR-10 dataset using Machine Learning

INSTRUCTIONS TO RUN THE BEST MODEL:
1.	The training binary files and the test binary file should be inside a folder called ‘dataset’. 
2.	batches.meta.txt should also be inside ‘dataset’ folder.
3.	The training binary files should be named as follows:
a)	data_batch_1.bin
b)	data_batch_2.bin
c)	data_batch_3.bin
d)	data_batch_4.bin
e)	data_batch_5.bin
4.	The test binary file should be names as follows:
a.	test_batch.bin
5.	The .py files should be in the same level as the ‘dataset’ folder.

 
 
RUNNING THE PYTHON FILE:
Python version 3.6 or 3.7
BEST MODEL: MODEL 1 HOG + SVM CLASSIFIER
Filename : model_1_hogSvm.py
It should be run from Anaconda Prompt with an argument (‘preTrain’ or ‘forceTrain’).
Example: 
 
ForceTrain – Starts Training from scratch
PreTrain – Loads previously trained classifier from folder if found and uses it to predict the test data.


2nd BEST MODEL : MODEL 2 VGG + RFC CLASSIFIER
Filename : model_2_vggRFC.py
Similar instructions as model 1
 
OUTPUT:
The testing Accuracy is displayed on the console.
The predictions are saved as a .csv file in the folder in respective model names.

NOTE:
If the pre-trained files are deleted from the folder, ‘ForceTrain’ is enforced automatically.
