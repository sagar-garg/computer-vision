#include <opencv2/opencv.hpp>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <random>
#include <stdlib.h>
#include "RandomForest.h"

using namespace std;

vector<vector<pair<int, cv::Mat>>> loadTask3Data()
{
    vector<pair<int, cv::Mat>> TrainingImages;
    vector<pair<int, cv::Mat>> TestImages;
    TrainingImages.reserve(53 + 81 + 51 + 290);
    TestImages.reserve(44);
    int numTrainImages[4] = {53, 81, 51, 290};
    int numTestImages[1] = {44};

    for (int i = 0; i < 4; i++)
    {
        for (size_t j = 0; j < numTrainImages[i]; j++)
        {
            stringstream imagePath;
            imagePath <<  "/home/madhan/Desktop/3rd_sem/TDCV/homework2/data/task3/train/" << setfill('0') << setw(2) << i << "/" << setfill('0') << setw(4) << j << ".jpg";
            string imagePathStr = imagePath.str();
            pair<int, cv::Mat> TrainImageWithLabelPair;
            TrainImageWithLabelPair.first = i;
            TrainImageWithLabelPair.second = imread(imagePathStr, cv::IMREAD_UNCHANGED).clone();
            TrainingImages.push_back(TrainImageWithLabelPair);
        }
    }

    for (size_t j = 0; j < numTestImages[0]; j++)
    {
        stringstream imagePath;
        imagePath <<  "/home/madhan/Desktop/3rd_sem/TDCV/homework2/data/task3/test/" << setfill('0') << setw(4) << j << ".jpg";
        string imagePathStr = imagePath.str();
        pair<int, cv::Mat> TestImageWithLabelPair;
        TestImageWithLabelPair.first = -1; // These test images have no label
        TestImageWithLabelPair.second = imread(imagePathStr, cv::IMREAD_UNCHANGED).clone();
        TestImages.push_back(TestImageWithLabelPair);
    }

    vector<vector<pair<int, cv::Mat>>> Dataset;
    Dataset.push_back(TrainingImages);
    Dataset.push_back(TestImages);
    return Dataset;
}

vector<vector<vector<int>>> getGTBoundingBoxesWithLabel()
{
    int numTestImages = 44;
    vector<vector<vector<int>>> LabelAndBoundingBoxes;
    for (size_t j = 0; j < numTestImages; j++)
    {
        stringstream Path;
        Path << "/home/madhan/Desktop/3rd_sem/TDCV/homework2/data/task3/gt/" << setfill('0') << setw(4) << j << ".gt.txt";
        string PathStr = Path.str();

        fstream GTfile;
        GTfile.open(PathStr);
        

        std::string line;
        vector<vector<int>> LabelAndBoundingBoxesPerImage;
        while (std::getline(GTfile, line))
        {
            std::istringstream buffer(line);
            vector<int> LabelAndBoundingBox(5);
            int temp;
            for (size_t i = 0; i < 5; i++)
            {
                buffer >> temp;
                LabelAndBoundingBox.at(i) = temp;
            }
            LabelAndBoundingBoxesPerImage.push_back(LabelAndBoundingBox);
        }
        LabelAndBoundingBoxes.push_back(LabelAndBoundingBoxesPerImage);
        GTfile.close();
    }
    return LabelAndBoundingBoxes;
}

void computeTask3(cv::Ptr<RandomForest> &randomForest, vector<pair<int, cv::Mat>> &testImagesLabelVector,
                vector<vector<vector<int>>> &GTlabelAndBoundingBoxes, cv::Size blockStep, 
                cv::Size padding, cv::Scalar *BoundingBoxColours, string outputDir)
{   

    float NMS_MAX_IOU_THRESHOLD = 0.3f; // If above this threshold, merge the two bounding boxes.
    
    float NMS_CONFIDENCE_THRESHOLD = 0.7f; // Predicted bounding boxes less than this confidence are removed
    int strideX = 2;
    int strideY = 2;

    for (size_t i = 0; i < testImagesLabelVector.size(); i++)
    {
        cout << "Running prediction on " << (i+1) << " of " << testImagesLabelVector.size() << " images.\n"; 
        
        cv::Mat testImage = testImagesLabelVector.at(i).second;

        //Get the maximum bounding box side length from the Ground truth boxes of image 
        int minBoundingBoxSideLength = 1000, maxBoundingBoxSideLength = -1;
        vector<vector<int>> imageLabelsAndBoundingBoxes = GTlabelAndBoundingBoxes.at(i);
       
        for (size_t j = 0; j < imageLabelsAndBoundingBoxes.size(); j++)
        {
            vector<int> bbox = imageLabelsAndBoundingBoxes.at(j);
            cv::Rect rect(bbox[1], bbox[2], bbox[3] - bbox[1], bbox[4] - bbox[2]);
            minBoundingBoxSideLength = min(minBoundingBoxSideLength, min(rect.width, rect.height));
            maxBoundingBoxSideLength = max(maxBoundingBoxSideLength, max(rect.width, rect.height));
        }
        

        int boundingBoxSideLength = (minBoundingBoxSideLength + maxBoundingBoxSideLength)/2;
        vector<Prediction> predictions; 
    
        //Predict objects  
        for (size_t row = 0; row < testImage.rows - boundingBoxSideLength; row += strideY)
        {
            for (size_t col = 0; col < testImage.cols - boundingBoxSideLength; col += strideX)
            {
                cv::Rect rect(col, row, boundingBoxSideLength, boundingBoxSideLength);
                cv::Mat rectImage = testImage(rect);
                
                Prediction prediction = randomForest->predict(rectImage, blockStep, padding);
                if (prediction.label != 3) // Ignore Background class.
                {
                    if(prediction.confidence > NMS_CONFIDENCE_THRESHOLD)    // Taking only bounding boxes with good confidence
                        prediction.bbox = rect;
                        predictions.push_back(prediction);
                }
            }
        }


        vector<Prediction> predictionsNMS;
        predictionsNMS.reserve(20); 

        for (auto &&prediction : predictions)
        {
            bool clusterFound = false;
            for (auto &&nmsCluster : predictionsNMS)
            {
                if (nmsCluster.label == prediction.label)
                { // Only if same label
                    Rect &rect1 = prediction.bbox;
                    Rect &rect2 = nmsCluster.bbox;
                    float iouScore = ((rect1 & rect2).area() * 1.0f) / ((rect1 | rect2).area());
                    if (iouScore > NMS_MAX_IOU_THRESHOLD) 
                    {   
                        nmsCluster.bbox = rect1 | rect2; // Merge the two bounding boxes
                        nmsCluster.confidence = max(prediction.confidence, nmsCluster.confidence);
                        clusterFound = true;
                        break;
                    }
                   
                }
            }
            if (!clusterFound)
                predictionsNMS.push_back(prediction);
        }
        // Draw predicted bounding boxes on the test image
        cv::Mat testImageCopy = testImage.clone(); 
        for (auto &&prediction : predictionsNMS)
            cv::rectangle(testImageCopy, prediction.bbox, BoundingBoxColours[prediction.label]);

        stringstream modelOutputFilePath;
        modelOutputFilePath << outputDir << setfill('0') << setw(4) << i << "-ModelOutput.png";
        string modelOutputFilePathStr = modelOutputFilePath.str();
        cv::imwrite(modelOutputFilePathStr, testImageCopy);

        // Draw GT bounding boxes on the test image
        imageLabelsAndBoundingBoxes = GTlabelAndBoundingBoxes.at(i);
        cv::Mat testImageGtCopy = testImage.clone(); 
        for (size_t j = 0; j < imageLabelsAndBoundingBoxes.size(); j++)
        {
            vector<int> bbox = imageLabelsAndBoundingBoxes.at(j);
            cv::Rect rect(bbox[1], bbox[2], bbox[3] - bbox[1], bbox[4] - bbox[2]);
            cv::rectangle(testImageGtCopy, rect, BoundingBoxColours[bbox[0]]);
        }

        stringstream gtFilePath;
        gtFilePath << outputDir << setfill('0') << setw(4) << i << "-GrountTruth.png";
        string gtFilePathStr = gtFilePath.str();
        cv::imwrite(gtFilePathStr, testImageGtCopy);      
    }
  
}

void task3()
{
    // Load all the images
    vector<vector<pair<int, cv::Mat>>> dataset = loadTask3Data();
    // Load the ground truth bounding boxes with their label values
    vector<vector<vector<int>>> GTlabelAndBoundingBoxes = getGTBoundingBoxesWithLabel();
    vector<pair<int, cv::Mat>> trainingImagesWithLabel = dataset.at(0);

    // Create model
    int numClasses = 4;
    int numDTrees = 60;
    Size winSize(128, 128);
    cv::Ptr<RandomForest> randomForest = RandomForest::create(numClasses, numDTrees, winSize);

    // Train the model
    Size blockStep(8, 8);
    Size padding(0, 0);
    float subsetPercentage = 60.0f;
    bool undersampling = true;
    bool augment = true;
    randomForest->train(trainingImagesWithLabel, subsetPercentage, blockStep, padding, undersampling, augment);

    vector<pair<int, cv::Mat>> testImagesWithLabel = dataset.at(1);
    cv::Scalar BoundingBoxColours[4];
    BoundingBoxColours[0] = cv::Scalar(255, 0, 0);   // label 0   Trimmer
    BoundingBoxColours[1] = cv::Scalar(0, 255, 0);   // label 1   Camera
    BoundingBoxColours[2] = cv::Scalar(0, 0, 255);   // label 2   Toothset
    BoundingBoxColours[3] = cv::Scalar(255, 255, 0); // label 3 background 

    std::ostringstream ss;
    
    string folderName = "predictions";
    string folderCreateCommand = "mkdir " + folderName;

    system(folderCreateCommand.c_str());

    ss<<folderName<<"/";

    string outputDir = ss.str();
    
    computeTask3(randomForest, testImagesWithLabel, GTlabelAndBoundingBoxes, blockStep, padding, BoundingBoxColours, outputDir);
               
}

int main()
{
    task3();
    return 0;
}
