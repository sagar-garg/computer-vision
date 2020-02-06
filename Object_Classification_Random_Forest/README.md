# TDCV Homework2

To install opencv in your home directory, follow this video https://www.youtube.com/watch?v=6pABIQl1ZP0

Some modifications:
1) change "python3.5-dev" to "python3.7-dev" and remove "libjasper-dev"
2) if you got error: No pakage opencv found

try this one:
sudo apt-get install libopencv-dev



#### To build the solution

For example for task1
to build the code:  g++ task1.cpp HOGDescriptor.cpp  -o output `pkg-config --cflags --libs opencv`

Implementation results

Decision tree of Accuracy 53.33

Random forest accuracy - somewhere around 80 are obtained

### Source

I referred completely on https://github.com/saurabheights/TDCV/tree/master/Exercise04-06 ----- Saviour for life :)


#### Task left: task3