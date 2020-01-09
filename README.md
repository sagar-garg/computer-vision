#computer vision
Assignments for Tracking and Detection in Computer Vision Course at Technical University of Munich

Implementations:
1. Image processing and HOG (Histogram of Oriented Gradients) descriptors
2. Object Classification (Random Forest)
3. Object Detection (Random Forest)

Steps:
To install opencv in your home directory, follow this video https://www.youtube.com/watch?v=6pABIQl1ZP0
Some modifications:

change "python3.5-dev" to "python3.7-dev" and remove "libjasper-dev"
if you got error: No pakage opencv found

try this one:
sudo apt-get install libopencv-dev

Running on Mac/Windows? - Think of DOCKER!

To build the solution
For example for task1
to build the code:  g++ task1.cpp HOGDescriptor.cpp  -o output pkg-config --cflags --libs opencv
Implementation results
Decision tree of Accuracy 53.33
Random forest accuracy - somewhere around 80 are obtained
