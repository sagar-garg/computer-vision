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
  int numberOfDTrees;
  int numberOfClasses;
  Size winSize;
  HOGDescriptor hog;
  vector<cv::Ptr<cv::ml::DTrees>> models;
  std::mt19937 randomGenerator;

  RandomForest();

  vector<int> getRandomUniqueIndices(int start, int end, int numOfSamples);
  HOGDescriptor createHogDescriptor();
  cv::Ptr<cv::ml::DTrees> trainDecisionTree(vector<pair<int, cv::Mat>> &trainingImagesLabelVector,
                                            Size blockStep,
                                            Size padding);
  cv::Mat resizeToBoundingBox(cv::Mat &inputImage);
  vector<pair<int, cv::Mat>>
  generateTrainingImagesLabelSubsetVector(vector<pair<int, cv::Mat>> &trainingImagesLabelVector,
                                          float subsetPercentage,
                                          bool undersampling);
  vector<cv::Mat> augmentImage(cv::Mat &inputImage);

public:
  static cv::Ptr<RandomForest> create(int numberOfClasses,
                                      int numberOfDTrees,
                                      Size winSize);
  void train(vector<pair<int, cv::Mat>> &trainingImagesLabelVector,
             float subsetPercentage,
             Size blockStep,
             Size padding,
             bool undersampling,
             bool augment);
  Prediction predict(cv::Mat &testImage,
                     Size blockStep,
                     Size padding);
};

#endif // RANDOM_FOREST_H