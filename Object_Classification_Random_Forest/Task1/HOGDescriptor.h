

#ifndef RF_HOGDESCRIPTOR_H
#define RF_HOGDESCRIPTOR_H


#include <opencv2/opencv.hpp>
#include <vector>

class HOGDescriptor {

public:

    HOGDescriptor() {
        //initialize default parameters(win_size, block_size, block_step,....)
        win_size = cv::Size(128, 128);
        
        //Fill other parameters here
        block_size = cv::Size(16, 16);
        block_step = cv::Size(8, 8);
        cell_size = cv::Size(8, 8);
        pad_size = cv::Size(0, 0);
        // parameter to check if descriptor is already initialized or not
        is_init = false;
    };


    void setWinSize(cv::Size sz) {
        //Fill
        win_size = sz;
    }

    cv::Size getWinSize(){
        //Fill
        return win_size;
    }

    void setBlockSize(cv::Size sz) {
        //Fill
        block_size = sz;
    }

    cv::Size getBlockSize(){
        //Fill
        return block_size;
    }

    void setBlockStep(cv::Size sz) {
       //Fill
       block_step = sz;
    }

    cv::Size getBlockStep(){
        //Fill
        return block_step;
    }

    void setCellSize(cv::Size sz) {
      //Fill
      cell_size = sz;
    }

    cv::Size getCellSize(){
        //Fill
        return cell_size;
    }

    void setPadSize(cv::Size sz) {
        pad_size = sz;
    }

    cv::Size getPadSize(){
        //Fill
        return pad_size;
    }

    void initDetector();

    void visualizeHOG(cv::Mat img, std::vector<float> &feats, cv::HOGDescriptor hog_detector, int scale_factor);

    void detectHOGDescriptor(cv::Mat &im);

    ~HOGDescriptor() {};


private:
    cv::Size win_size;

    /*
        Fill other parameters here
    */
    cv::Size block_size;
    cv::Size block_step;
    cv::Size cell_size;
    cv::Size pad_size;
    cv::HOGDescriptor hog_detector;
    bool is_init;
};

#endif //RF_HOGDESCRIPTOR_H
