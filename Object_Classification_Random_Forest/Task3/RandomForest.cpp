#include "RandomForest.h"
#include <algorithm> // For std::shuffle
#include <iostream>  // For std::cout, std::endl
#include <random>    // For std::mt19937, std::random_device
#include <vector>    // For std::vector
#include <iterator>
#include <opencv2/opencv.hpp>
using namespace std;
using namespace cv;

RandomForest::RandomForest(/* args */)
{
}

cv::Ptr<RandomForest> RandomForest::create(int numberOfClasses,
                                           int numberOfDTrees,
                                           Size winSize)
{
    cv::Ptr<RandomForest> randomForest = new RandomForest();
    randomForest->numberOfClasses = numberOfClasses;
    randomForest->numberOfDTrees = numberOfDTrees;
    randomForest->winSize = winSize;
    randomForest->models.reserve(numberOfDTrees);
    randomForest->hog = randomForest->createHogDescriptor();
    long unsigned int timestamp = static_cast<long unsigned int>(time(0));
    randomForest->randomGenerator = std::mt19937(timestamp);
    return randomForest;
}

vector<int> RandomForest::getRandomUniqueIndices(int start, int end, int numOfSamples)
{
    std::vector<int> indices;
    indices.reserve(end - start);
    for (size_t i = start; i < end; i++)
        indices.push_back(i);

    std::shuffle(indices.begin(), indices.end(), randomGenerator);
    // copy(indices.begin(), indices.begin() + numOfSamples, std::ostream_iterator<int>(std::cout, ", "));
    // cout << endl;
    return std::vector<int>(indices.begin(), indices.begin() + numOfSamples);
}

vector<cv::Mat> RandomForest::augmentImage(cv::Mat &inputImage)
{
    vector<cv::Mat> augmentations;
    cv::Mat currentImage = inputImage;
    cv::Mat rotatedImage, flippedImage;
    for (size_t j = 0; j < 4; j++)
    {
        if (j == 0)
        {
            rotatedImage = currentImage;
            augmentations.push_back(rotatedImage);
        }
        else
        {
            cv::rotate(currentImage, rotatedImage, cv::ROTATE_90_CLOCKWISE);
            augmentations.push_back(rotatedImage);
        }

        for (int i = 0; i <= 1; i++)
        {
            cv::flip(rotatedImage, flippedImage, i);
            augmentations.push_back(flippedImage);
            
        }
        currentImage = rotatedImage;
    }
    
    return augmentations;
}

vector<pair<int, cv::Mat>> RandomForest::generateTrainingImagesLabelSubsetVector(vector<pair<int, cv::Mat>> &trainingImagesLabelVector,
                                                                                 float subsetPercentage,
                                                                                 bool undersampling)
{

    vector<pair<int, cv::Mat>> trainingImagesLabelSubsetVector;

    // Compute minimum number of samples a class label has.
    int minimumSample = trainingImagesLabelVector.size(); // A high enough value

    if (undersampling)
    {
        int minimumClassSamples[numberOfClasses];
        for (size_t i = 0; i < numberOfClasses; i++)
            minimumClassSamples[i] = 0;
        for (auto &&trainingSample : trainingImagesLabelVector)
            minimumClassSamples[trainingSample.first]++;
        for (size_t i = 1; i < numberOfClasses; i++)
            if (minimumClassSamples[i] < minimumSample)
                minimumSample = minimumClassSamples[i];
    }

    for (size_t label = 0; label < numberOfClasses; label++)
    {
        // Create a subset vector for all the samples with class label.
        vector<pair<int, cv::Mat>> temp;
        temp.reserve(100);
        for (auto &&sample : trainingImagesLabelVector)
            if (sample.first == label)
                temp.push_back(sample);

        // Compute how many samples to choose for each label for random subset.
        int numOfElements;
        if (undersampling)
        {
            numOfElements = (subsetPercentage * minimumSample) / 100;
        }
        else
        {
            numOfElements = (temp.size() * subsetPercentage) / 100;
        }

        // Filter numOfElements elements from temp and append to trainingImagesLabelSubsetVector
        vector<int> randomUniqueIndices = getRandomUniqueIndices(0, temp.size(), numOfElements);
        for (size_t j = 0; j < randomUniqueIndices.size(); j++)
        {
            pair<int, cv::Mat> subsetSample = temp.at(randomUniqueIndices.at(j));
            trainingImagesLabelSubsetVector.push_back(subsetSample);
        }
    }

    return trainingImagesLabelSubsetVector;
}

void RandomForest::train(vector<pair<int, cv::Mat>> &trainingImagesLabelVector,
                         float subsetPercentage,
                         Size blockStep,
                         Size padding,
                         bool undersampling,
                         bool augment)
{
    // Augment the dataset
    vector<pair<int, cv::Mat>> augmentedTrainingImagesLabelVector;
    augmentedTrainingImagesLabelVector.reserve(trainingImagesLabelVector.size() * 12);
    if (augment)
    {
        for(auto&& trainingImagesLabelSample : trainingImagesLabelVector)
        {
            vector<cv::Mat> augmentedImages = augmentImage(trainingImagesLabelSample.second);
            for (auto &&augmentedImage : augmentedImages)
            {
                augmentedTrainingImagesLabelVector.push_back(pair<int, cv::Mat>(trainingImagesLabelSample.first, augmentedImage));
            }
        }
    } else {
        augmentedTrainingImagesLabelVector = trainingImagesLabelVector;
    }

    // Train each decision tree
    for (size_t i = 0; i < numberOfDTrees; i++)
    {
        cout << "Training decision tree: " << i + 1 << " of " << numberOfDTrees << ".\n";
        vector<pair<int, cv::Mat>> trainingImagesLabelSubsetVector =
            generateTrainingImagesLabelSubsetVector(augmentedTrainingImagesLabelVector,
                                                    subsetPercentage,
                                                    undersampling);

        cv::Ptr<cv::ml::DTrees> model = trainDecisionTree(trainingImagesLabelSubsetVector,
                                                          blockStep,
                                                          padding);
        models.push_back(model);
    }
}

Prediction RandomForest::predict(cv::Mat &testImage,
                                 Size blockStep,
                                 Size padding)
{
    cv::Mat resizedInputImage = resizeToBoundingBox(testImage);

    // Compute Hog only of center crop of grayscale image
    vector<float> descriptors;
    vector<Point> foundLocations;
    vector<double> weights;

    cv::Mat grayImage;
    cv::cvtColor(resizedInputImage, grayImage, cv::COLOR_BGR2GRAY);

    hog.compute(grayImage, descriptors, blockStep, padding, foundLocations);

    // Store the features and labels for model training.
    // cout << i << ": Expected: " << testImagesLabelVector.at(i).first << ", Found: " << model->predict(cv::Mat(descriptors)) << endl ;
    // if(testImagesLabelVector.at(i).first == randomForest.at(0)->predict(cv::Mat(descriptors)))
    //     accuracy += 1;
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
                      .confidence = (labelCounts[maxCountLabel] * 1.0f) / numberOfDTrees};
}

cv::Ptr<cv::ml::DTrees> RandomForest::trainDecisionTree(vector<pair<int, cv::Mat>> &trainingImagesLabelVector,
                                                        Size blockStep,
                                                        Size padding)
{
    // Create the model
    cv::Ptr<cv::ml::DTrees> model = cv::ml::DTrees::create();
    // See https://docs.opencv.org/3.0-beta/modules/ml/doc/decision_trees.html#dtrees-params
    model->setCVFolds(0);        // set num cross validation folds - Not implemented in OpenCV
    model->setMaxCategories(10); // set max number of categories
    model->setMaxDepth(20);      // set max tree depth
    model->setMinSampleCount(2); // set min sample count
    
    // Compute Hog Features for all the training images
    cv::Mat feats, labels;
    for (size_t i = 0; i < trainingImagesLabelVector.size(); i++)
    {
        cv::Mat inputImage = trainingImagesLabelVector.at(i).second;
        cv::Mat resizedInputImage = resizeToBoundingBox(inputImage);

        // Compute Hog only of center crop of grayscale image
        vector<float> descriptors;
        vector<Point> foundLocations;
        vector<double> weights;

        cv::Mat grayImage;
        cv::cvtColor(resizedInputImage, grayImage, cv::COLOR_BGR2GRAY);

        hog.compute(grayImage, descriptors, blockStep, padding, foundLocations);

        // Store the features and labels for model training.
        // cout << "=====================================" << endl;
        // cout << "Number of descriptors are: " << descriptors.size() << endl;
        feats.push_back(cv::Mat(descriptors).clone().reshape(1, 1));
        // cout << "New size of training features" << feats.size() << endl;
        labels.push_back(trainingImagesLabelVector.at(i).first);
        // cout << "New size of training labels" << labels.size() << endl;
    }

    cv::Ptr<cv::ml::TrainData> trainData = ml::TrainData::create(feats, ml::ROW_SAMPLE, labels);
    model->train(trainData);
    return model;
}

HOGDescriptor RandomForest::createHogDescriptor()
{
    // Create Hog Descriptor
    Size blockSize(16, 16);
    Size blockStride(8, 8);
    Size cellSize(8, 8);
    int nbins(18);
    
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