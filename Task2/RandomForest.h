#ifndef RANDOM_FOREST_H
#define RANDOM_FOREST_H

#include <vector>
#include <opencv2/opencv.hpp>
#include <utility>
#include <random> // For std::mt19937, std::random_device
using namespace std;
using namespace cv;

struct Prediction
{
  int label;
  float confidence;
  cv::Rect bbox;
};

class RandomForest
{
private:
  int num_trees;
  int num_classes;
  Size winSize;
  HOGDescriptor hog;
  vector<cv::Ptr<cv::ml::DTrees>> models;

  vector<int> getRandomIndices(int start, int end, int num_samples);
  HOGDescriptor createHogDescriptor();
  cv::Ptr<cv::ml::DTrees> trainDecisionTree(vector<pair<int, cv::Mat>> &trainingset,
                                            Size blockStep,
                                            Size padding);
  cv::Mat resizeToBoundingBox(cv::Mat &inputImage);
  vector<pair<int, cv::Mat>> genTrainingSubset(vector<pair<int, cv::Mat>> &trainingset,
                                          float subsetPercent,
                                          bool undersampling);
  vector<cv::Mat> augmentImage(cv::Mat &inputImage);

public:
  static cv::Ptr<RandomForest> create(int num_classes,
                                      int num_trees,
                                      Size winSize);
  void train(vector<pair<int, cv::Mat>> &trainingset,
             float subsetPercent,
             Size blockStep,
             Size padding,
             bool undersampling,
             bool augment);
  Prediction predict(cv::Mat &testImage,
                     Size blockStep,
                     Size padding);
};

#endif // RANDOM_FOREST_H
