#include <opencv2/highgui.hpp>
#include <opencv2/tracking.hpp>
#include <opencv2/tracking/tracking_legacy.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/tracking.hpp>
#include <iostream>
#include <vector>
#include <cmath>
#include <pthread.h>
#include "serialLib/serialib.h"

#define TUNE 
// #define SERIAL

using namespace cv;
using namespace std;

#define NUM_VAR 1
#define HEADER 0x12
#define SERIAL_PORT "/dev/ttyUSB0"
#define BUFFER_SIZE (1+(NUM_VAR*4))
#define try
uint8_t sendBuf[BUFFER_SIZE];

int main(int argc, char **argv)
{
	bool die(false);

	Mat rgbMat(Size(640, 480), CV_8UC3, Scalar(0));

	VideoCapture cap(14, CAP_V4L2); //capture the video from webcam
	namedWindow("rgb", WINDOW_AUTOSIZE);
	//RED
	int iLowH = 116, iLowS = 154, iLowV = 42;
	int iHighH = 179, iHighS = 255, iHighV = 255;
	//BLUE
	int bLowH = 0, bLowS = 190, bLowV = 67;
	int bHighH = 92, bHighS = 255, bHighV = 206;

	Mat imgOriginal, imgHSV, imgHSVblue, imgThresholded, imgThresholdedBlue;
	vector<Vec4i> hierarchy;
	int centerx = 320, centery = 240, numRed, numBlue, totalDetected;

	cv::Ptr<cv::legacy::TrackerMedianFlow> tracker;

#ifdef TUNE
	namedWindow("Control", WINDOW_AUTOSIZE); //create a window called "Control"
	//Create trackbars in "Control" window
	createTrackbar("LowH", "Control", &iLowH, 179); //Hue (0 - 179)
	createTrackbar("HighH", "Control", &iHighH, 179);

	createTrackbar("LowS", "Control", &iLowS, 255); //Saturation (0 - 255)
	createTrackbar("HighS", "Control", &iHighS, 255);

	createTrackbar("LowV", "Control", &iLowV, 255); //Value (0 - 255)
	createTrackbar("HighV", "Control", &iHighV, 255);

	createTrackbar("BLowH", "Control", &bLowH, 179); //Hue (0 - 179)
	createTrackbar("BHighH", "Control", &bHighH, 179);

	createTrackbar("BLowS", "Control", &bLowS, 255); //Saturation (0 - 255)
	createTrackbar("BHighS", "Control", &bHighS, 255);

	createTrackbar("BLowV", "Control", &bLowV, 255); //Value (0 - 255)
	createTrackbar("BHighV", "Control", &bHighV, 255);
#endif

#ifdef SERIAL
	serialib serial;
    char errorOpening = serial.openDevice(SERIAL_PORT, 115200);
    if (errorOpening!=1) {
        printf("Haven't or cannot connect to COM\n");
        return errorOpening;
    }
    printf ("Successful connection to %s\n",SERIAL_PORT);
	sendBuf[0] = HEADER;
#endif
	while (!die)
	{
		cap.read(rgbMat);
		flip(rgbMat, rgbMat, 0);
		cvtColor(rgbMat, imgHSV, COLOR_RGB2HSV); //Convert the captured frame from BGR to HSV

		inRange(imgHSV, Scalar(iLowH, iLowS, iLowV), Scalar(iHighH, iHighS, iHighV), imgThresholded); //Threshold the image
		inRange(imgHSV, Scalar(bLowH, bLowS, bLowV), Scalar(bHighH, bHighS, bHighV), imgThresholdedBlue);
		//morphological opening (removes small objects from the foreground)
		erode(imgThresholded, imgThresholded, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)));
		dilate(imgThresholded, imgThresholded, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)));
		erode(imgThresholdedBlue, imgThresholdedBlue, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)));
		dilate(imgThresholdedBlue, imgThresholdedBlue, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)));
		//morphological closing (removes small holes from the foreground)
		dilate(imgThresholded, imgThresholded, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)));
		erode(imgThresholded, imgThresholded, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)));
		dilate(imgThresholdedBlue, imgThresholdedBlue, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)));
		erode(imgThresholdedBlue, imgThresholdedBlue, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)));

		// Canny(imgThresholded, canny_output, 50, 150, 3);
		vector<vector<Point>> contours, contoursBlue, allContours, filteredRed, filteredBlue;

		findContours(imgThresholded, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE, Point(0, 0));
		// Canny(imgThresholdedBlue, canny_output, 50, 150, 3);
		findContours(imgThresholdedBlue, contoursBlue, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE, Point(0, 0));
		
		vector<Moments> mu;
		vector<Point2f> mc;
		for(int i=0; i<contours.size(); i++){
			Moments moment = moments(contours[i], false);
			if(moment.m00 > 2800.0){//area
				// contours.erase(contours.begin() + i-back);
				filteredRed.push_back(contours[i]);
				mu.push_back(moment);
				mc.push_back(Point2f(mu[i].m10 / mu[i].m00, mu[i].m01 / mu[i].m00));
			}
		}
		numRed = filteredRed.size();
		for(int i=0; i<contoursBlue.size(); i++){
			Moments moment = moments(contoursBlue[i], false);
			if(moment.m00 > 2800.0){
				// contoursBlue.erase(contoursBlue.begin() + i-back);
				filteredBlue.push_back(contoursBlue[i]);
				mu.push_back(moment);
				mc.push_back(Point2f(moment.m10 / moment.m00, moment.m01 / moment.m00));
			}				
		}	
		numBlue = filteredBlue.size();

		allContours.insert(allContours.begin(), filteredRed.begin(), filteredRed.end());
		allContours.insert(allContours.end(), filteredBlue.begin(), filteredBlue.end());

		Scalar color;
		printf("Red = %d Blue = %d\n", filteredRed.size(), filteredBlue.size());
		vector<Rect> allRect(allContours.size());
		totalDetected = allContours.size();
		for (int i = 0; i < allContours.size(); i++)
		{
			if (i < numRed){
				color = Scalar(0, 0, 255); // B G R values
			}
			else{
				color = Scalar(255, 0, 0);
			}
			// printf("mc %.3f %.3f\n", mc[i].x, mc[i].y);
			drawContours(rgbMat, allContours, i, color, 2, 8, hierarchy, 0, Point());
			// Rect borect = boundingRect(allContours[i]);
			// allRect.push_back(borect); // x y w h
			// putText(imgOriginal, to_string(allContours[i].size()), mc[i], FONT_HERSHEY_COMPLEX, 1, color, 1, 8, false);
			// putText(imgOriginal, to_string(i) + "nd", Point(mc[i].x, mc[i].y - borect.height), FONT_HERSHEY_COMPLEX, 1, color, 1, 8, false);
			// rectangle(imgOriginal, borect, color, 1, 8, 0);
			int D_X = mc[i].x - centerx;
			circle(rgbMat, mc[i], 4, color, -1, 8, 0);
			putText(rgbMat, "Area: " + to_string((int)mu[i].m00), Point(mc[i].x - 50, mc[i].y - 100), FONT_HERSHEY_COMPLEX, 0.5, color, 1.5, 8, 0);
			putText(rgbMat, "D_X: " + to_string(D_X), Point(mc[i].x - 50, mc[i].y - 60), FONT_HERSHEY_COMPLEX, 0.5, color, 1.5, 8, false);
			putText(rgbMat, to_string(i), Point(mc[i].x, mc[i].y - 10), FONT_HERSHEY_COMPLEX, 0.5, Scalar(0, 255, 0), 1.5, 8, false);
			circle(rgbMat, Point(centerx, centery), 4, Scalar(0, 255, 0), -1, 8, 0);
			printf("i:%d, A:%d, D_X:%d\n", i, (int)mu[i].m00, D_X);
			if(!isnan(mc[i].x) && !isnan(mc[i].y)){
#ifdef SERIAL
				memcpy(&sendBuf[1], &D_X, 4);
				serial.writeBytes(sendBuf, BUFFER_SIZE);
#endif
			}
		}
		cv::imshow("rgb", rgbMat);
#ifdef TUNE
		imshow("Thresholded Image Red", imgThresholded);		//show the thresholded image
		imshow("Thresholded Image Blue", imgThresholdedBlue); 	//show the thresholded image
		moveWindow("Thresholded Image Red", 700, 30);
#endif
#ifdef SERIAL
		char command;
		serial.readChar(&command, 1);
#endif
		char key = cv::waitKey(5);
		if (key == 27)
		{
			cv::destroyWindow("rgb");
			break;
		}
	}
	return 0;
}
