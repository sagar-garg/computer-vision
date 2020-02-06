
#include <opencv2/opencv.hpp>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <random>
#include <stdlib.h>
#include "RandomForest.h"

using namespace std;


vector<vector<pair<int, cv::Mat>>> loadTask2Dataset()
{
    vector<pair<int, cv::Mat>> TrainingImages;
    vector<pair<int, cv::Mat>> TestImages;
    TrainingImages.reserve(49 + 67 + 42 + 53 + 67 + 110); //Total number of training images for all 6 classes
    TestImages.reserve(60);    //Total number of test images for all 6 classes
    int numberOfTrainingImages[6] = {49, 67, 42, 53, 67, 110};
    int numberOfTestImages[6] = {10, 10, 10, 10, 10, 10};

    for (int i = 0; i < 6; i++)
    {
        for (size_t j = 0; j < numberOfTrainingImages[i]; j++)
        {
            stringstream imagePath;
            imagePath << "/home/madhan/Desktop/3rd_sem/TDCV/homework2/data/task2/train/" << setfill('0') << setw(2) << i << "/" << setfill('0') << setw(4) << j << ".jpg";
            string imagePathStr = imagePath.str();
            pair<int, cv::Mat> labelImagesTrainPair;
            labelImagesTrainPair.first = i;    // label of image
            labelImagesTrainPair.second = imread(imagePathStr, cv::IMREAD_UNCHANGED).clone();
            TrainingImages.push_back(labelImagesTrainPair);
        }

        for (size_t j = 0; j < numberOfTestImages[i]; j++)
        {
            stringstream imagePath;
            imagePath << "/home/madhan/Desktop/3rd_sem/TDCV/homework2/data/task2/test/" << setfill('0') << setw(2) << i << "/" << setfill('0') << setw(4) << j + numberOfTrainingImages[i] << ".jpg";
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
    cv::Mat resizedInputImage;
    if (inputImage.rows < winSize.height || inputImage.cols < winSize.width)
    {
        float scaleFactor = fmax((winSize.height * 1.0f) / inputImage.rows, (winSize.width * 1.0f) / inputImage.cols);
        cv::resize(inputImage, resizedInputImage, cv::Size(0, 0), scaleFactor, scaleFactor, cv::INTER_LINEAR);
    }
    else
    {
        resizedInputImage = inputImage;
    }

    cv::Rect r = cv::Rect((resizedInputImage.cols - winSize.width) / 2, (resizedInputImage.rows - winSize.height) / 2,
                  winSize.width, winSize.height);
    return resizedInputImage(r);
}

cv::Ptr<cv::ml::DTrees> trainDecisionTree(vector<pair<int, cv::Mat>> &trainingImagesLabelVector)
{
    // Create the model
    cv::Ptr<cv::ml::DTrees> decisionTreeModel = cv::ml::DTrees::create();
    decisionTreeModel->setCVFolds(0); // set num cross validation folds 
    decisionTreeModel->setMaxCategories(10);
    decisionTreeModel->setMaxDepth(10);       // set max tree depth
    decisionTreeModel->setMinSampleCount(2); // set min sample count
    cout << "Number of cross validation folds are: " << decisionTreeModel->getCVFolds() << endl;
    cout << "Max Categories are: " << decisionTreeModel->getMaxCategories() << endl;
    cout << "Max depth is: " << decisionTreeModel->getMaxDepth() << endl;
    cout << "Minimum Sample Count: " << decisionTreeModel->getMinSampleCount() << endl;
    
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
    vector<vector<pair<int, cv::Mat>>> dataset = loadTask2Dataset();
    vector<pair<int, cv::Mat>> trainingImagesLabelVector = dataset.at(0);

    // Train model
    cv::Ptr<cv::ml::DTrees> decisionTreeModel = trainDecisionTree(trainingImagesLabelVector);

    // Predict on test dataset
    vector<pair<int, cv::Mat>> testImagesLabelVector = dataset.at(1);
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

    for (size_t i = 0; i < testImagesLabelVector.size(); i++)
    {
        cv::Mat inputImage = testImagesLabelVector.at(i).second;
        cv::Mat resizedInputImage = resizeToBoundingBox(inputImage, winSize);

        // Compute Hog only of center crop of grayscale image
        vector<float> descriptors;
        vector<cv::Point> foundLocations;
        vector<double> weights;

        cv::Mat grayImage;
        cv::cvtColor(resizedInputImage, grayImage, cv::COLOR_BGR2GRAY);

        hog.compute(grayImage, descriptors, blockStep, padding, foundLocations);

        
        if (testImagesLabelVector.at(i).first == decisionTreeModel->predict(cv::Mat(descriptors)))
            count += 1;
    }

    cout << "==================================================" << endl;
    cout << "TASK 2 - Accuracy of Single Decision tree is: [" << (count / testImagesLabelVector.size())*100.0f << "]." << endl;
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
    vector<vector<pair<int, cv::Mat>>> dataset = loadTask2Dataset();
    vector<pair<int, cv::Mat>> trainingImagesLabelVector = dataset.at(0);

    // Create model
    int numberOfClasses = 6;
    int numberOfDTrees = 50;
    cv::Size winSize(128, 128);
    cv::Ptr<RandomForest> randomForest = RandomForest::create(numberOfClasses, numberOfDTrees, winSize);

    // Train the model
    cv::Size blockStep(8, 8);
    cv::Size padding(0, 0);
    float subsetPercentage = 50.0f;
    bool undersampling = true;
    bool augment = false;
    randomForest->train(trainingImagesLabelVector, subsetPercentage, blockStep, padding, undersampling, augment);

    // Predict on test dataset
    vector<pair<int, cv::Mat>> testImagesLabelVector = dataset.at(1);
    float count = 0;
    float countPerClass[6] = {0};
    for (size_t i = 0; i < testImagesLabelVector.size(); i++)
    {
        cv::Mat testImage = testImagesLabelVector.at(i).second;
        Prediction prediction = randomForest->predict(testImage, blockStep, padding);
        if (testImagesLabelVector.at(i).first == prediction.label)
        {
            count += 1;
            countPerClass[prediction.label] += 1;
        }
    }

    cout << "==================================================" << endl;
    cout << "TASK 2 - Accuracy of Random Forest is: [" << (count / testImagesLabelVector.size())*100.0f << "]." << endl;

    int numberOfTestImages[6] = {10, 10, 10, 10, 10, 10};
    for (size_t i = 0; i < numberOfClasses; i++)
    {
        cout << "Accuracy of Class " << i << " : [" << (countPerClass[i] / numberOfTestImages[i])*100.0f << "]." << endl;
    }
    cout << "==================================================" << endl;

}

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


void task3_core(cv::Ptr<RandomForest> &randomForest,
                vector<pair<int, cv::Mat>> &testImagesLabelVector,
                vector<vector<vector<int>>> &labelAndBoundingBoxes,
                int strideX, int strideY,
                cv::Size winStride, cv::Size padding,
                cv::Scalar *gtColors,
                float scaleFactor,
                string outputDir)
{   

    // NMS-Not used. Each boundin box is dumped to the text file which contains the confidence value. The thresholding is done in evaluation.cpp
    float NMS_MAX_IOU_THRESHOLD = 0.5f; // If above this threshold, merge the two bounding boxes.
    float NMS_MIN_IOU_THRESHOLD = 0.1f; // If above this threshold, drop the bounding boxes with lower confidence.
    float NMS_CONFIDENCE_THRESHOLD = 0.8f;
    
    ofstream predictionsFile(outputDir + "predictions.txt");
    if (!predictionsFile.is_open())
    {
        cout << "Failed to open" << outputDir + "predictions.txt" << endl;
        exit(-1);
    }

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
                    else if (iouScore > NMS_MIN_IOU_THRESHOLD) // ToDo: Improve this.
                    {
                    //     // Drop the bounding box with lower confidence
                         if (nmsCluster.confidence < prediction.confidence)
                         {
                             nmsCluster = prediction;
                         }
                         clusterFound = true;
                         break;
                    }
                   
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
    }
    predictionsFile.close();
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
    int numberOfDTrees = 50;
    Size winSize(128, 128);
    cv::Ptr<RandomForest> randomForest = RandomForest::create(numberOfClasses, numberOfDTrees, winSize);

    // Train the model
    Size winStride(8, 8);
    Size padding(0, 0);
    float subsetPercentage = 50.0f;
    bool undersampling = false;
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
    task3_core(randomForest, testImagesLabelVector, labelAndBoundingBoxes, strideX, strideY, winStride, padding, gtColors,
               scaleFactor, outputDir);
               
}

int main(){
    testDTrees();
    testForest();
    task3();
    return 0;
}
