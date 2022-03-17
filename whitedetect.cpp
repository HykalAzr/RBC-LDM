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

// #define TUNE 
#define SERIAL

using namespace cv;
using namespace std;

#define NUM_VAR 1
#define HEADER 0x12
#define TAIL 0x21
#define SERIAL_PORT "/dev/ttyUSB2"
#define BUFFER_SIZE (1+(NUM_VAR*4)+1)
#define try
uint8_t sendBuf[BUFFER_SIZE];

int main(int argc, char **argv)
{
	bool die(false);

	Mat rgbMat(Size(640, 480), CV_8UC3, Scalar(0));

	VideoCapture cap(14, CAP_V4L2);                 //capture the video from webcam
	namedWindow("rgb", WINDOW_AUTOSIZE);
	//RED
	int iLowH = 0, iLowS = 0, iLowV = 230;
	int iHighH = 179, iHighS = 35, iHighV = 255;

	Mat imgOriginal, imgHSV, imgThresholded;
	vector<Vec4i> hierarchy;
	int centerx = 320, centery = 240, numWhite, numClosest, prevDX;

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
	sendBuf[BUFFER_SIZE - 1] = TAIL;
#endif
	while (!die)
	{
		cap.read(rgbMat);
		flip(rgbMat, rgbMat, 0);
		cvtColor(rgbMat, imgHSV, COLOR_RGB2HSV); //Convert the captured frame from BGR to HSV

		inRange(imgHSV, Scalar(iLowH, iLowS, iLowV), Scalar(iHighH, iHighS, iHighV), imgThresholded); //Threshold the image
		//morphological opening (removes small objects from the foreground)
		erode(imgThresholded, imgThresholded, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)));
		dilate(imgThresholded, imgThresholded, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)));
		//morphological closing (removes small holes from the foreground)
		dilate(imgThresholded, imgThresholded, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)));
		erode(imgThresholded, imgThresholded, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)));

		// Canny(imgThresholded, canny_output, 50, 150, 3);
		vector<vector<Point>> contours, allContours, filteredWhite;

		findContours(imgThresholded, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE, Point(0, 0));
		
		vector<Moments> mu;
		vector<Point2f> mc;
		for(int i=0; i<contours.size(); i++){
			Moments moment = moments(contours[i], false);
			if(moment.m00 > 2800.0){//area
				// contours.erase(contours.begin() + i-back);
				filteredWhite.push_back(contours[i]);
				mu.push_back(moment);
				mc.push_back(Point2f(mu[i].m10 / mu[i].m00, mu[i].m01 / mu[i].m00));
			}
		}
		numWhite = filteredWhite.size();

		allContours.insert(allContours.begin(), filteredWhite.begin(), filteredWhite.end());

		Scalar color;
		printf("White = %d\n", filteredWhite.size());
		vector<Rect> allRect(allContours.size());
        prevDX = 1000;
		for (int i = 0; i < allContours.size(); i++)
		{
			if (i < numWhite){
				color = Scalar(0, 0, 255); // B G R values
			}
			else{
				color = Scalar(255, 0, 0);
			}
			drawContours(rgbMat, allContours, i, color, 2, 8, hierarchy, 0, Point());
			int D_X = mc[i].x - centerx;
			circle(rgbMat, mc[i], 4, color, -1, 8, 0);
			putText(rgbMat, "Area: " + to_string((int)mu[i].m00), Point(mc[i].x - 50, mc[i].y - 100), FONT_HERSHEY_COMPLEX, 0.5, color, 1.5, 8, 0);
			putText(rgbMat, "D_X: " + to_string(D_X), Point(mc[i].x - 50, mc[i].y - 60), FONT_HERSHEY_COMPLEX, 0.5, color, 1.5, 8, false);
			putText(rgbMat, to_string(i), Point(mc[i].x, mc[i].y - 10), FONT_HERSHEY_COMPLEX, 0.5, Scalar(0, 255, 0), 1.5, 8, false);
			circle(rgbMat, Point(centerx, centery), 4, Scalar(0, 255, 0), -1, 8, 0);
			printf("i:%d, A:%d, D_X:%d\n", i, (int)mu[i].m00, D_X);
            if(D_X < 0)
                D_X = -(D_X);

            if(D_X < prevDX)
                numClosest = i;

            prevDX = D_X;
		}
		sendBuf[0] = HEADER;
		sendBuf[BUFFER_SIZE - 1] = TAIL;
        if(filteredWhite.size() != 0){
			int sendArea = (int)mu[numClosest].m00;
		    if(!isnan(mc[numClosest].x) && !isnan(mc[numClosest].y)){
#ifdef SERIAL
				memcpy(&sendBuf[1], &sendArea, 4);
				serial.writeBytes(sendBuf, BUFFER_SIZE);
                printf("Aserial: %d\n", sendArea);
#endif
            }
            numClosest = 0;
        }
		cv::imshow("rgb", rgbMat);
#ifdef TUNE
		imshow("Thresholded Image Red", imgThresholded);		//show the thresholded image
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