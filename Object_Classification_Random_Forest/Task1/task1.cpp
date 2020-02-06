
#include <opencv2/opencv.hpp>

#include "HOGDescriptor.h"




int main(){
    cv::Mat im = cv::imread("/home/madhan/Desktop/3rd_sem/TDCV/homework2/data/task1/obj1000.jpg");
    cv::imshow("task1 - Input Image", im);
    cv::waitKey(-1);

    //Image converted from BGR to Gray scale
    cv::Mat grayImage;
    cv::cvtColor(im, grayImage, cv::COLOR_BGR2GRAY);
    cv::imshow("Input Image in Gray scale", grayImage);
    cv::waitKey(-1);

    //Image resized to 75% of input image
    cv::Mat resizedInputImage;
    cv::resize(im, resizedInputImage, cv::Size(im.cols * 0.75,im.rows * 0.75), 0, 0, cv::INTER_LINEAR);
    cv::imshow("Resized Input Image (75%)", resizedInputImage);
    cv::waitKey(-1);

    //Rotate image by -45 degree
    double angle = -45;
    // get rotation matrix for rotating the image around its center in pixel coordinates
    cv::Point2f center((im.cols-1)/2.0, (im.rows-1)/2.0);
    cv::Mat rot = cv::getRotationMatrix2D(center, angle, 1.0);
    // determine bounding rectangle, center not relevant
    cv::Rect2f bbox = cv::RotatedRect(cv::Point2f(), im.size(), angle).boundingRect2f();
    // adjust transformation matrix
    rot.at<double>(0,2) += bbox.width/2.0 - im.cols/2.0;
    rot.at<double>(1,2) += bbox.height/2.0 - im.rows/2.0;

    cv::Mat dst;
    cv::warpAffine(im, dst, rot, bbox.size());
    cv::imshow("Rotated Input Image (-45 deg)", dst);
    cv::waitKey(-1);
    

    //Flip image along y axis
    cv::Mat flippedImage;
    cv::flip(im,flippedImage,1);
    cv::imshow("Flipped Input Image (along y axis)", flippedImage);
    cv::waitKey(-1);

	//Fill Code here

    /*
    	* Create instance of HOGDescriptor and initialize
    	* Compute HOG descriptors
    	* visualize
    */
    HOGDescriptor hog_detector;
    hog_detector.initDetector();
    hog_detector.detectHOGDescriptor(im);

    return 0;
}