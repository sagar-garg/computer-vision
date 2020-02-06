#include <opencv2/opencv.hpp>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <random>
#include <stdlib.h>
#include "RandomForest.h"

using namespace std;

vector<vector<pair<int, cv::Mat>>> loadTask3Dataset()
{
    vector<pair<int, cv::Mat>> TrainingImages;
    vector<pair<int, cv::Mat>> TestImages;
    TrainingImages.reserve(53 + 81 + 51 + 290);
    TestImages.reserve(44);
    int numberOfTrainImages[4] = {53, 81, 51, 290};
    int numberOfTestImages[1] = {44};

    for (int i = 0; i < 4; i++)
    {
        for (size_t j = 0; j < numberOfTrainImages[i]; j++)
        {
            stringstream imagePath;
            imagePath <<  "/home/madhan/Desktop/3rd_sem/TDCV/homework2/data/task3/train/" << setfill('0') << setw(2) << i << "/" << setfill('0') << setw(4) << j << ".jpg";
            string imagePathStr = imagePath.str();
            // cout << imagePathStr << endl;
            pair<int, cv::Mat> labelImagesTrainPair;
            labelImagesTrainPair.first = i;
            labelImagesTrainPair.second = imread(imagePathStr, cv::IMREAD_UNCHANGED).clone();
            TrainingImages.push_back(labelImagesTrainPair);
        }
    }

    for (size_t j = 0; j < numberOfTestImages[0]; j++)
    {
        stringstream imagePath;
        imagePath <<  "/home/madhan/Desktop/3rd_sem/TDCV/homework2/data/task3/test/" << setfill('0') << setw(4) << j << ".jpg";
        string imagePathStr = imagePath.str();
        // cout << imagePathStr << endl;
        pair<int, cv::Mat> labelImagesTestPair;
        labelImagesTestPair.first = -1; // These test images have no label
        labelImagesTestPair.second = imread(imagePathStr, cv::IMREAD_UNCHANGED).clone();
        TestImages.push_back(labelImagesTestPair);
    }

    vector<vector<pair<int, cv::Mat>>> Dataset;
    Dataset.push_back(TrainingImages);
    Dataset.push_back(TestImages);
    return Dataset;
}

vector<vector<vector<int>>> getLabelAndBoundingBoxes()
{
    int numberOfTestImages = 44;
    vector<vector<vector<int>>> LabelAndBoundingBoxes;
    for (size_t j = 0; j < numberOfTestImages; j++)
    {
        stringstream gtPath;
        gtPath << "/home/madhan/Desktop/3rd_sem/TDCV/homework2/data/task3/gt/" << setfill('0') << setw(4) << j << ".gt.txt";
        string gtPathStr = gtPath.str();

        fstream gtFile;
        gtFile.open(gtPathStr);
        

        std::string line;
        vector<vector<int>> LabelAndBoundingBoxesPerImage;
        while (std::getline(gtFile, line))
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
        gtFile.close();
    }
    return LabelAndBoundingBoxes;
}


vector<float> computeTpFpFn(vector<Prediction> predictionsNMSVector,
                            vector<Prediction> groundTruthPredictions)
{
    float tp = 0, fp = 0, fn = 0;
    float matchThresholdIou = 0.3f;

    for (auto &&myPrediction : predictionsNMSVector)
    {
        bool matchesWithAnyGroundTruth = false;
        Rect myRect = myPrediction.bbox;

        for (auto &&groundTruth : groundTruthPredictions)
        {
            if (groundTruth.label != myPrediction.label)
                continue;
            Rect gtRect = groundTruth.bbox;
            float iouScore = ((myRect & gtRect).area() * 1.0f) / ((myRect | gtRect).area());
            if (iouScore > matchThresholdIou)
            {
                matchesWithAnyGroundTruth = true;
                break;
            }
        }

        if (matchesWithAnyGroundTruth)
            tp++;
        else
            fp++;
    }

    for (auto &&groundTruth : groundTruthPredictions)
    {
        bool isGtBboxMissed = true;
        Rect gtRect = groundTruth.bbox;
        for (auto &&myPrediction : predictionsNMSVector)
        {
            if (groundTruth.label != myPrediction.label)
                continue;
            Rect myRect = myPrediction.bbox;
            float iouScore = ((myRect & gtRect).area() * 1.0f) / ((myRect | gtRect).area());
            if (iouScore > matchThresholdIou)
            {
                isGtBboxMissed = false;
                break;
            }
        }

        if (isGtBboxMissed)
            fn++;
    }

    vector<float> results;
    results.push_back(tp);
    results.push_back(fp);
    results.push_back(fn);
    return results;
}


vector<float> task3_core(cv::Ptr<RandomForest> &randomForest,
                vector<pair<int, cv::Mat>> &testImagesLabelVector,
                vector<vector<vector<int>>> &labelAndBoundingBoxes,
                int strideX, int strideY,
                cv::Size winStride, cv::Size padding,
                cv::Scalar *gtColors,
                float scaleFactor,
                string outputDir, float NMS_CONFIDENCE_THRESHOLD)
{   

    // NMS-Not used. Each boundin box is dumped to the text file which contains the confidence value. The thresholding is done in evaluation.cpp
    float NMS_MAX_IOU_THRESHOLD = 0.5f; // If above this threshold, merge the two bounding boxes.
    float NMS_MIN_IOU_THRESHOLD = 0.1f; // If above this threshold, drop the bounding boxes with lower confidence.
    
    
    ofstream predictionsFile(outputDir + "predictions.txt");
    

    float tp = 0, fp = 0, fn = 0;
    for (size_t i = 0; i < testImagesLabelVector.size(); i++)
    {
        cout << "Running prediction on " << (i+1) << " of " << testImagesLabelVector.size() << " images.\n"; 
        predictionsFile << i << endl; // Prediction file format: Starts with File number
        cv::Mat testImage = testImagesLabelVector.at(i).second;

        // Run testing on various bounding boxes of different scales
        // int minBoundingBoxSideLength = 70, maxBoundingBoxSideLength = 230;
        int minBoundingBoxSideLength = 1000, maxBoundingBoxSideLength = -1;
        vector<vector<int>> imageLabelsAndBoundingBoxes = labelAndBoundingBoxes.at(i);
        predictionsFile << imageLabelsAndBoundingBoxes.size() << endl; // Prediction file format: Next is Number of Ground Truth Boxes - Say K
        for (size_t j = 0; j < imageLabelsAndBoundingBoxes.size(); j++)
        {
            vector<int> bbox = imageLabelsAndBoundingBoxes.at(j);
            cv::Rect rect(bbox[1], bbox[2], bbox[3] - bbox[1], bbox[4] - bbox[2]);
            // Prediction file format: Next is K Lines of Labels and cv::Rect
            predictionsFile << imageLabelsAndBoundingBoxes.at(j).at(0) << " " << rect.x << " " << rect.y << " " << rect.height << " " << rect.width << endl;
            minBoundingBoxSideLength = min(minBoundingBoxSideLength, min(rect.width, rect.height));
            maxBoundingBoxSideLength = max(maxBoundingBoxSideLength, max(rect.width, rect.height));
        }
        minBoundingBoxSideLength -= 10;
        maxBoundingBoxSideLength += 10;

        int boundingBoxSideLength = minBoundingBoxSideLength;
        vector<Prediction> predictionsVector; // Output of Hog Detection
        while (true)
        {
            cout << "Processing at bounding box side length: " << boundingBoxSideLength << '\n';
            // Sliding window with stride
            for (size_t row = 0; row < testImage.rows - boundingBoxSideLength; row += strideY)
            {
                for (size_t col = 0; col < testImage.cols - boundingBoxSideLength; col += strideX)
                {
                    cv::Rect rect(col, row, boundingBoxSideLength, boundingBoxSideLength);
                    cv::Mat rectImage = testImage(rect);

                    // Predict on subimage
                    Prediction prediction = randomForest->predict(rectImage, winStride, padding);
                    if (prediction.label != 3) // Ignore Background class.
                    {
                        if(prediction.confidence > NMS_CONFIDENCE_THRESHOLD)    // Taking only bounding boxes with good confidence
                            prediction.bbox = rect;
                            predictionsVector.push_back(prediction);
                    }
                }
            }

            if (boundingBoxSideLength == maxBoundingBoxSideLength) // Maximum Bounding Box Size from ground truth
                break;
            boundingBoxSideLength = (boundingBoxSideLength * scaleFactor + 0.5);
            if (boundingBoxSideLength > maxBoundingBoxSideLength)
                boundingBoxSideLength = maxBoundingBoxSideLength;
        }

        vector<Prediction> predictionsNMSVector;
        predictionsNMSVector.reserve(20); 

        for (auto &&prediction : predictionsVector)
        {
           
            // Check if NMS already has a cluster which shares NMS_IOU_THRESHOLD area with current prediction.bbox and both have same label.
            bool clusterFound = false;
            for (auto &&nmsCluster : predictionsNMSVector)
            {
                if (nmsCluster.label == prediction.label)
                { // Only if same label
                    Rect &rect1 = prediction.bbox;
                    Rect &rect2 = nmsCluster.bbox;
                    float iouScore = ((rect1 & rect2).area() * 1.0f) / ((rect1 | rect2).area());
                    if (iouScore > NMS_MAX_IOU_THRESHOLD) // Merge the two bounding boxes
                    {
                        nmsCluster.bbox = rect1 | rect2;
                        nmsCluster.confidence = max(prediction.confidence, nmsCluster.confidence);
                        clusterFound = true;
                        break;
                    }
                   // else if (iouScore > NMS_MIN_IOU_THRESHOLD) // ToDo: Improve this.
                   // {
                   // //     // Drop the bounding box with lower confidence
                   //      if (nmsCluster.confidence < prediction.confidence)
                   //      {
                   //          nmsCluster = prediction;
                   //      }
                   //      clusterFound = true;
                   //      break;
                   // }
                   
                }
            }

            // If no NMS cluster found, add the prediction as a new cluster
            if (!clusterFound)
                predictionsNMSVector.push_back(prediction);
        }



        // Prediction file format: Next is N Lines of Labels, cv::Rect and confidence
        predictionsFile << predictionsNMSVector.size() << endl;
        for (auto &&prediction : predictionsNMSVector)
        {
            // Prediction file format: Next is N Lines of Labels and cv::Rect
            predictionsFile << prediction.label << " " << prediction.bbox.x << " " << prediction.bbox.y << " " << prediction.bbox.height << " " << prediction.bbox.width << " " << prediction.confidence << endl;
        }

        cv::Mat testImageClone = testImage.clone(); // For drawing bbox
        for (auto &&prediction : predictionsNMSVector)
            cv::rectangle(testImageClone, prediction.bbox, gtColors[prediction.label]);

        // Draw bounding box on the test image using ground truth
        imageLabelsAndBoundingBoxes = labelAndBoundingBoxes.at(i);
        cv::Mat testImageGtClone = testImage.clone(); // For drawing bbox
        for (size_t j = 0; j < imageLabelsAndBoundingBoxes.size(); j++)
        {
            vector<int> bbox = imageLabelsAndBoundingBoxes.at(j);
            cv::Rect rect(bbox[1], bbox[2], bbox[3] - bbox[1], bbox[4] - bbox[2]);
            cv::rectangle(testImageGtClone, rect, gtColors[bbox[0]]);
        }

        stringstream modelOutputFilePath;
        modelOutputFilePath << outputDir << setfill('0') << setw(4) << i << "-ModelOutput.png";
        string modelOutputFilePathStr = modelOutputFilePath.str();
        cv::imwrite(modelOutputFilePathStr, testImageClone);

        stringstream gtFilePath;
        gtFilePath << outputDir << setfill('0') << setw(4) << i << "-GrountTruth.png";
        string gtFilePathStr = gtFilePath.str();
        cv::imwrite(gtFilePathStr, testImageGtClone);

        vector<Prediction> groundTruthPredictions;
        for (size_t j = 0; j < 3; j++)
        {
            Prediction groundTruthPrediction;
            groundTruthPrediction.label = labelAndBoundingBoxes.at(i).at(j).at(0);
            groundTruthPrediction.bbox.x = labelAndBoundingBoxes.at(i).at(j).at(1);
            groundTruthPrediction.bbox.y = labelAndBoundingBoxes.at(i).at(j).at(2);
            groundTruthPrediction.bbox.height = labelAndBoundingBoxes.at(i).at(j).at(3);
            groundTruthPrediction.bbox.height -= groundTruthPrediction.bbox.x;
            groundTruthPrediction.bbox.width = labelAndBoundingBoxes.at(i).at(j).at(4);
            groundTruthPrediction.bbox.width -= groundTruthPrediction.bbox.y;
            groundTruthPredictions.push_back(groundTruthPrediction);
        }

        vector<float> tpFpFn = computeTpFpFn(predictionsNMSVector, groundTruthPredictions);

        tp += tpFpFn[0];
        fp += tpFpFn[1];
        fn += tpFpFn[2];
    }
    predictionsFile.close();

    float precision = tp / (tp + fp);
    float recall = tp / (tp + fn);
    predictionsFile.close();
    vector<float> precisionRecallValue;
    precisionRecallValue.push_back(precision);
    precisionRecallValue.push_back(recall);
    return precisionRecallValue;
}

void task3()
{
    // Load all the images
    vector<vector<pair<int, cv::Mat>>> dataset = loadTask3Dataset();
    // Load the ground truth bounding boxes with their label values
    vector<vector<vector<int>>> labelAndBoundingBoxes = getLabelAndBoundingBoxes();
    vector<pair<int, cv::Mat>> trainingImagesLabelVector = dataset.at(0);

    // Create model
    int numberOfClasses = 4;
    int numberOfDTrees = 60;
    Size winSize(128, 128);
    cv::Ptr<RandomForest> randomForest = RandomForest::create(numberOfClasses, numberOfDTrees, winSize);

    // Train the model
    Size winStride(8, 8);
    Size padding(0, 0);
    float subsetPercentage = 60.0f;
    bool undersampling = true;
    bool augment = true;
    randomForest->train(trainingImagesLabelVector, subsetPercentage, winStride, padding, undersampling, augment);

    // For each test image
    vector<pair<int, cv::Mat>> testImagesLabelVector = dataset.at(1);
    cv::Scalar gtColors[4];
    gtColors[0] = cv::Scalar(255, 0, 0);
    gtColors[1] = cv::Scalar(0, 255, 0);
    gtColors[2] = cv::Scalar(0, 0, 255);
    gtColors[3] = cv::Scalar(255, 255, 0);

    float scaleFactor = 1.20;
    int strideX = 2;
    int strideY = 2;
    
    

    // Loop over multiple values.
    std::ostringstream ss;
    //s << "/home/madhan/Desktop/3rd_sem/TDCV/homework2/output/Trees-" << numberOfDTrees << "_subsetPercent-" << ((int)subsetPercentage) << "-undersampling_" << undersampling << "-augment_" << augment << "-strideX_" << strideX << "-strideY_" << strideY << "-NMS_MIN_" << NMS_MIN_IOU_THRESHOLD << "-NMS_Max_" << NMS_MAX_IOU_THRESHOLD << "-NMS_CONF_" << NMS_CONFIDENCE_THRESHOLD << "/";
    //string outputDir = s.str();
    string folderName = "predictions";
    string folderCreateCommand = "mkdir " + folderName;

    system(folderCreateCommand.c_str());

    ss<<folderName<<"/";

    string outputDir = ss.str();

    ofstream outputFile;
    outputFile.open(outputDir+"predictionRecallValues.csv");
    
    outputFile << "Precision,Recall"<< endl;
    for (int confidence = 0; confidence <= 100; confidence += 5) // If float is used, it may overshoot 1.0 - floating point error
    {
        float NMS_CONFIDENCE_THRESHOLD = confidence / 100.0f;
        vector<float> precisionRecallValue = task3_core(randomForest, testImagesLabelVector, labelAndBoundingBoxes, strideX, strideY, winStride, padding, gtColors,
               scaleFactor, outputDir, NMS_CONFIDENCE_THRESHOLD);
        cout << "NMS_CONFIDENCE_THRESHOLD: " << NMS_CONFIDENCE_THRESHOLD << ", Precision: " << precisionRecallValue[0] << ", Recall: " << precisionRecallValue[1] << endl;
        outputFile << precisionRecallValue[0] << "," << precisionRecallValue[1] << endl;
    }
    outputFile.close();
    
               
}

int main()
{
    task3();
    return 0;
}