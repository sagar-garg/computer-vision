#include "RandomForest.h"
#include <algorithm> 
#include <iostream>  
#include <random>    
#include <vector>    
#include <iterator>
#include <opencv2/opencv.hpp>
using namespace std;
using namespace cv;

cv::Ptr<RandomForest> RandomForest::create(int num_classes,int num_trees,Size winSize)
{
    cv::Ptr<RandomForest> randomForest = new RandomForest();
    randomForest->num_classes = num_classes;
    randomForest->num_trees = num_trees;
    randomForest->winSize = winSize;
    randomForest->models.reserve(num_trees);
    randomForest->hog = randomForest->createHogDescriptor();
    return randomForest;
}

vector<cv::Mat> RandomForest::augmentImage(cv::Mat &inputImage)
{
    vector<cv::Mat> augmented;
    cv::Mat currentImage = inputImage;
    cv::Mat rotatedImage, flippedImage;
    for (size_t j = 0; j < 4; j++) //1 rotation.then 2 flips. total 3*4.
    {
        if (j == 0)
        {
            rotatedImage = currentImage;
            augmented.push_back(rotatedImage);
        }
        else
        {
            cv::rotate(currentImage, rotatedImage, cv::ROTATE_90_CLOCKWISE);
            augmented.push_back(rotatedImage);
        }

        for (int i = 0; i <= 1; i++)
        {
            cv::flip(rotatedImage, flippedImage, i);
            augmented.push_back(flippedImage);
            
        }
        currentImage = rotatedImage;
    }
    
    return augmented;
}

vector<int> RandomForest::getRandomIndices(int start, int end, int num_samples)
{
    std::vector<int> indices;
    std::random_device rd;
    std::mt19937 g(rd());
 
    indices.reserve(end - start);
    for (size_t i = start; i < end; i++)
        indices.push_back(i);

    std::shuffle(indices.begin(), indices.end(), g); //returns a shuffled array of size (end-begin)
    return std::vector<int>(indices.begin(), indices.begin() + num_samples); //takes first n entries from shuffled array
}

vector<pair<int, cv::Mat>> RandomForest::genTrainingSubset(vector<pair<int, cv::Mat>> &trainingset,
                                                                                 float subsetPercent,
                                                                                 bool undersampling)
{
    vector<pair<int, cv::Mat>> trainingSubset;
    int minimum_sample_size = trainingset.size();

    if (undersampling)
    {//the minimum number of samples among different classes. (here 50*12)
        int minimum_sample_class[num_classes];
        for (size_t i = 0; i < num_classes; i++)
            minimum_sample_class[i] = 0; //initializing to zero.
        for (auto &&trainingSample : trainingset)
            minimum_sample_class[trainingSample.first]++; //each entry now contains total samples in that class.
        for (size_t i = 1; i < num_classes; i++)
            if (minimum_sample_class[i] < minimum_sample_size)
                minimum_sample_size = minimum_sample_class[i]; //minimum among all classes.
    }

    for (size_t label = 0; label < num_classes; label++)
    { 
        vector<pair<int, cv::Mat>> temp;
        temp.reserve(3500);//max images=289*12.
        for (auto &&sample : trainingset)
            if (sample.first == label)
                temp.push_back(sample);

        int num_elements;
        if (undersampling)
        {
            num_elements = (subsetPercent * minimum_sample_size) / 100;
        }
        else
        {
            num_elements = (temp.size() * subsetPercent) / 100;
        }

        // Filter num_elements elements from temp and append to trainingSubset
        vector<int> randomIndices = getRandomIndices(0, temp.size(), num_elements);
        for (size_t j = 0; j < randomIndices.size(); j++)
        {
            pair<int, cv::Mat> subsetSample = temp.at(randomIndices.at(j));
            trainingSubset.push_back(subsetSample);
        }
    }

    return trainingSubset;
}

void RandomForest::train(vector<pair<int, cv::Mat>> &trainingset,
                         float subsetPercent,
                         Size blockStep,
                         Size padding,
                         bool undersampling,
                         bool augment)
{
    // Augment the dataset
    vector<pair<int, cv::Mat>> augmentedTrainingSet;
    augmentedTrainingSet.reserve(trainingset.size() * 12);//rotate, flip twice.
    if (augment)
    {
        for(auto&& trainingSample : trainingset)
        {
            vector<cv::Mat> augmentedImages = augmentImage(trainingSample.second);
            for (auto &&augmentedImage : augmentedImages)
            {
                augmentedTrainingSet.push_back(pair<int, cv::Mat>(trainingSample.first, augmentedImage));
            }
        }
    } else {
        augmentedTrainingSet = trainingset;
    }

    // Train each decision tree
    for (size_t i = 0; i < num_trees; i++)
    {
        cout << "Training decision tree: " << i + 1 << " of " << num_trees << ".\n";
        vector<pair<int, cv::Mat>> trainingSubset =
            genTrainingSubset(augmentedTrainingSet,
                                                    subsetPercent,
                                                    undersampling);

        cv::Ptr<cv::ml::DTrees> model = trainDecisionTree(trainingSubset,
                                                          blockStep,
                                                          padding);
        models.push_back(model);
    }
}

Prediction RandomForest::predict(cv::Mat &testImage, Size blockStep, Size padding)
{
    cv::Mat resizedInputImage = resizeToBoundingBox(testImage);

    vector<float> descriptors;
    vector<Point> foundLocations;
    vector<double> weights;

    cv::Mat grayImage;
    cv::cvtColor(resizedInputImage, grayImage, cv::COLOR_BGR2GRAY);

    hog.compute(grayImage, descriptors, blockStep, padding, foundLocations);

    std::map<int, int> labelCounts;
    int maxCountLabel = -1;
    for (auto &&model : models)
    {
        int label = model->predict(cv::Mat(descriptors));
        if (labelCounts.count(label) > 0)
            labelCounts[label]++;
        else
            labelCounts[label] = 1;

        if (maxCountLabel == -1)
            maxCountLabel = label;
        else if (labelCounts[label] > labelCounts[maxCountLabel])
            maxCountLabel = label;
    }

    return Prediction{.label = maxCountLabel,
                      .confidence = (labelCounts[maxCountLabel] * 1.0f) / num_trees};
}

cv::Ptr<cv::ml::DTrees> RandomForest::trainDecisionTree(vector<pair<int, cv::Mat>> &trainingset,
                                                        Size blockStep,
                                                        Size padding)
{
    // model creation
    cv::Ptr<cv::ml::DTrees> model = cv::ml::DTrees::create();
    model->setCVFolds(0);        
    model->setMaxCategories(10); 
    model->setMaxDepth(20); //should be high     
    model->setMinSampleCount(2);
    
    // hog computing
    cv::Mat feats, labels;
    for (size_t i = 0; i < trainingset.size(); i++)
    {
        cv::Mat inputImage = trainingset.at(i).second;
        cv::Mat resizedInputImage = resizeToBoundingBox(inputImage);

        vector<float> descriptors;
        vector<Point> foundLocations;
        vector<double> weights;

        cv::Mat grayImage;
        cv::cvtColor(resizedInputImage, grayImage, cv::COLOR_BGR2GRAY);

        hog.compute(grayImage, descriptors, blockStep, padding, foundLocations);

        feats.push_back(cv::Mat(descriptors).clone().reshape(1, 1));
        labels.push_back(trainingset.at(i).first);
    }

    cv::Ptr<cv::ml::TrainData> trainData = ml::TrainData::create(feats, ml::ROW_SAMPLE, labels);
    model->train(trainData);
    return model;
}

HOGDescriptor RandomForest::createHogDescriptor()
{
    Size blockSize(16, 16);
    Size blockStride(8, 8);
    Size cellSize(8, 8);
    int nbins(18); //doubled.
    
    HOGDescriptor hog(winSize, blockSize, blockStride, cellSize, nbins);
    return hog;
}

cv::Mat RandomForest::resizeToBoundingBox(cv::Mat &inputImage)
{
    cv::Mat resizedInputImage;
    if (inputImage.rows < winSize.height || inputImage.cols < winSize.width)
    {
        float scaleFactor = fmax((winSize.height * 1.0f) / inputImage.rows, (winSize.width * 1.0f) / inputImage.cols);
        cv::resize(inputImage, resizedInputImage, Size(0, 0), scaleFactor, scaleFactor, cv::INTER_LINEAR);
    }
    else
    {
        resizedInputImage = inputImage;
    }

    Rect r = Rect((resizedInputImage.cols - winSize.width) / 2, (resizedInputImage.rows - winSize.height) / 2,
                  winSize.width, winSize.height);
    
    return resizedInputImage(r);
}
