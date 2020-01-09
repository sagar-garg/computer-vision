
#include <opencv2/opencv.hpp>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <random>
#include <stdlib.h>
#include "RandomForest.h"

using namespace std;


vector<vector<pair<int, cv::Mat>>> loadTask2Data()
{
    vector<pair<int, cv::Mat>> TrainingImages;
    vector<pair<int, cv::Mat>> TestImages;
    TrainingImages.reserve(49 + 67 + 42 + 53 + 67 + 110); //Total number of training images for all 6 classes
    TestImages.reserve(60);    //Total number of test images for all 6 classes
    int numOfTrainingImages[6] = {49, 67, 42, 53, 67, 110};
    int numOfTestImages[6] = {10, 10, 10, 10, 10, 10};

    for (int i = 0; i < 6; i++)
    {
        for (size_t j = 0; j < numOfTrainingImages[i]; j++)
        {
            stringstream imagePath;
            imagePath << "/home/madhan/Desktop/3rd_sem/TDCV/homework2/data/task2/train/" << setfill('0') << setw(2) << i << "/" << setfill('0') << setw(4) << j << ".jpg";
            string imagePathStr = imagePath.str();
            pair<int, cv::Mat> labelImagesTrainPair;
            labelImagesTrainPair.first = i;    // label of image
            labelImagesTrainPair.second = imread(imagePathStr, cv::IMREAD_UNCHANGED).clone();
            TrainingImages.push_back(labelImagesTrainPair);
        }

        for (size_t j = 0; j < numOfTestImages[i]; j++)
        {
            stringstream imagePath;
            imagePath << "/home/madhan/Desktop/3rd_sem/TDCV/homework2/data/task2/test/" << setfill('0') << setw(2) << i << "/" << setfill('0') << setw(4) << j + numOfTrainingImages[i] << ".jpg";
            string imagePathStr = imagePath.str();
            // cout << imagePathStr << endl;
            pair<int, cv::Mat> labelImagesTestPair;
            labelImagesTestPair.first = i;    // label of image
            labelImagesTestPair.second = imread(imagePathStr, cv::IMREAD_UNCHANGED).clone();
            TestImages.push_back(labelImagesTestPair);
        }
    }

    vector<vector<pair<int, cv::Mat>>> Dataset;
    Dataset.push_back(TrainingImages);
    Dataset.push_back(TestImages);
    return Dataset;
}



cv::Mat resizeToBoundingBox(cv::Mat &inputImage, cv::Size &winSize)
{
    cv::Mat resizedImage;
    if (inputImage.rows < winSize.height || inputImage.cols < winSize.width)
    {
        float scaleFactor = fmax((winSize.height * 1.0f) / inputImage.rows, (winSize.width * 1.0f) / inputImage.cols);
        cv::resize(inputImage, resizedImage, cv::Size(0, 0), scaleFactor, scaleFactor, cv::INTER_LINEAR);
    }
    else
    {
        resizedImage = inputImage;
    }

    cv::Rect r = cv::Rect((resizedImage.cols - winSize.width) / 2, (resizedImage.rows - winSize.height) / 2,
                  winSize.width, winSize.height);
    return resizedImage(r);
}

cv::Ptr<cv::ml::DTrees> trainDecisionTree(vector<pair<int, cv::Mat>> &trainingImagesLabelVector)
{
    // Create the model
    cv::Ptr<cv::ml::DTrees> decisionTreeModel = cv::ml::DTrees::create();
    decisionTreeModel->setCVFolds(0); // set num cross validation folds 
    decisionTreeModel->setMaxCategories(10);
    decisionTreeModel->setMaxDepth(20);       // set max tree depth
    decisionTreeModel->setMinSampleCount(2); // set min sample count
    
    // Compute Hog Features for all the training images
    cv::Size winSize(128, 128);
    cv::Size blockSize(16, 16);
    cv::Size blockStep(8, 8);
    cv::Size cellSize(8, 8);
    int nbins(9);
    int derivAperture(1);
    double winSigma(-1);
    int histogramNormType(HOGDescriptor::L2Hys);
    double L2HysThreshold(0.2);
    bool gammaCorrection(true);
    float free_coef(-1.f);
    //! Maximum number of detection window increases. Default value is 64
    int nlevels(HOGDescriptor::DEFAULT_NLEVELS);
    //! Indicates signed gradient will be used or not
    bool signedGradient(false);
    cv::HOGDescriptor hog(winSize, blockSize, blockStep, cellSize, nbins, derivAperture, winSigma, histogramNormType,
                      L2HysThreshold, gammaCorrection, nlevels, signedGradient);
 
    
    cv::Size padding(0, 0);
    
    
    cv::Mat feats, labels;
    for (size_t i = 0; i < trainingImagesLabelVector.size(); i++)
    {   

        cv::Mat inputImage = trainingImagesLabelVector.at(i).second;
        
       
        //Resizing input image
        cv::Mat resizedInputImage = resizeToBoundingBox(inputImage, winSize);
        

        // Compute Hog only of center crop of grayscale image
        vector<float> descriptors;
        vector<cv::Point> foundLocations;

        cv::Mat grayImage;
        cv::cvtColor(resizedInputImage, grayImage, cv::COLOR_BGR2GRAY);
        
        hog.compute(grayImage, descriptors, blockStep, padding, foundLocations);
        
        // Store the features and labels for model training.
        
        feats.push_back(cv::Mat(descriptors).clone().reshape(1, 1));
        
        labels.push_back(trainingImagesLabelVector.at(i).first);
        
    }
    
    cv::Ptr<cv::ml::TrainData> trainData = cv::ml::TrainData::create(feats, cv::ml::ROW_SAMPLE, labels);
    decisionTreeModel->train(trainData);
    
    return decisionTreeModel;
}

void testDTrees() {

    int num_classes = 6;

    /* 
      * Create your data (i.e Use HOG from task 1 to compute the descriptor for your images)
      * Train a single Decision Tree and evaluate the performance 
      * Experiment with the MaxDepth parameter, to see how it affects the performance

    */
     // Load all the images
    vector<vector<pair<int, cv::Mat>>> dataset = loadTask2Data();
    vector<pair<int, cv::Mat>> trainingImagesWithLabel = dataset.at(0);

    // Train model
    cv::Ptr<cv::ml::DTrees> decisionTreeModel = trainDecisionTree(trainingImagesWithLabel);

    // Predict on test dataset
    vector<pair<int, cv::Mat>> testImagesWithLabel = dataset.at(1);
    float count = 0;
    cv::Size winSize(128, 128);
    cv::Size blockSize(16, 16);
    cv::Size blockStep(8, 8);
    cv::Size cellSize(8, 8);

    int nbins(9);
    int derivAperture(1);
    double winSigma(-1);
    int histogramNormType(HOGDescriptor::L2Hys);
    double L2HysThreshold(0.2);
    bool gammaCorrection(true);
    float free_coef(-1.f);
    //! Maximum number of detection window increases. Default value is 64
    int nlevels(HOGDescriptor::DEFAULT_NLEVELS);
    //! Indicates signed gradient will be used or not
    bool signedGradient(false);
    cv::HOGDescriptor hog(winSize, blockSize, blockStep, cellSize, nbins, derivAperture, winSigma, histogramNormType,
                      L2HysThreshold, gammaCorrection, nlevels, signedGradient);
 
    cv::Size padding(0, 0);

    for (size_t i = 0; i < testImagesWithLabel.size(); i++)
    {
        cv::Mat inputImage = testImagesWithLabel.at(i).second;
        cv::Mat resizedInputImage = resizeToBoundingBox(inputImage, winSize);

        // Compute Hog only of center crop of grayscale image
        vector<float> descriptors;
        vector<cv::Point> foundLocations;
        vector<double> weights;

        cv::Mat grayImage;
        cv::cvtColor(resizedInputImage, grayImage, cv::COLOR_BGR2GRAY);

        hog.compute(grayImage, descriptors, blockStep, padding, foundLocations);

        
        if (testImagesWithLabel.at(i).first == decisionTreeModel->predict(cv::Mat(descriptors)))
            count += 1;
    }

    cout << "==================================================" << endl;
    cout << "TASK 2 - Accuracy of Single Decision tree is: [" << (count / testImagesWithLabel.size())*100.0f << "]." << endl;
    cout << "==================================================" << endl;


}


void testForest(){

    int num_classes = 6;

    /* 
      * 
      * Create your data (i.e Use HOG from task 1 to compute the descriptor for your images)
      * Train a Forest and evaluate the performance 
      * Experiment with the MaxDepth & TreeCount parameters, to see how it affects the performance

    */
    vector<vector<pair<int, cv::Mat>>> dataset = loadTask2Data();
    vector<pair<int, cv::Mat>> trainingImagesWithLabel = dataset.at(0);

    // Create model
    int numClasses = 6;
    int numDTrees = 60;
    cv::Size winSize(128, 128);
    cv::Ptr<RandomForest> randomForest = RandomForest::create(numClasses, numDTrees, winSize);

    // Train the model
    cv::Size blockStep(8, 8);
    cv::Size padding(0, 0);
    float subsetPercentage = 60.0f;
    bool undersampling = true;
    bool augment = false;                        //No augmentation for task 2
    randomForest->train(trainingImagesWithLabel, subsetPercentage, blockStep, padding, undersampling, augment);

    // Predict on test dataset
    vector<pair<int, cv::Mat>> testImagesWithLabel = dataset.at(1);
    float count = 0;
    float countPerClass[6] = {0};
    for (size_t i = 0; i < testImagesWithLabel.size(); i++)
    {
        cv::Mat testImage = testImagesWithLabel.at(i).second;
        Prediction prediction = randomForest->predict(testImage, blockStep, padding);
        if (testImagesWithLabel.at(i).first == prediction.label)
        {
            count += 1;
            countPerClass[prediction.label] += 1;
        }
    }

    cout << "==================================================" << endl;
    cout << "TASK 2 - Accuracy of Random Forest is: [" << (count / testImagesWithLabel.size())*100.0f << "]." << endl;

    int numTestImages[6] = {10, 10, 10, 10, 10, 10};
    for (size_t i = 0; i < numClasses; i++)
    {
        cout << "Accuracy of Class " << i << " : [" << (countPerClass[i] / numTestImages[i])*100.0f << "]." << endl;
    }
    cout << "==================================================" << endl;

}

int main(){
    testDTrees();
    testForest();
    return 0;
}
