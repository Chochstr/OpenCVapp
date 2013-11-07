#include "stdafx.h"
#include "opencv2/core/core.hpp"
#include "opencv2/contrib/contrib.hpp"
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/opencv.hpp"
#include <iostream>
#include <fstream>
#include <sstream>
#include <stdio.h>
#include <Windows.h>
#include <process.h>

using namespace std;
using namespace cv;
using namespace System::Threading;

CascadeClassifier haar, haar_cascade2, haar_cascade3, haar_cascade4, haar_cascade5, lbp_cascade;

// Haar
string cascade_alt = "C:/Users/Chochstr/My Programs/opencv/data/haarcascades/haarcascade_frontalface_alt.xml";
string cascade_alt2 = "C:/Users/Chochstr/My Programs/opencv/data/haarcascades/haarcascade_frontalface_alt2.xml";
string cascade_default = "C:/Users/Chochstr/My Programs/opencv/data/haarcascades/haarcascade_frontalface_default.xml";
string cascade_alt_tree = "C:/Users/Chochstr/My Programs/opencv/data/haarcascades/haarcascade_frontalface_alt_tree.xml";
string cascade_profile = "C:/Users/Chochstr/My Programs/opencv/data/haarcascades/haarcascade_profileface.xml";

// LBP
string lbpcascade_front ="C:/Users/Chochstr/My Programs/opencv/data/lbpcascades/lbpcascade_frontalface.xml";
string lbpcascade_profile ="C:/Users/Chochstr/My Programs/opencv/data/lbpcascades/lbpcascade_profileface.xml";

// Eyes
string cascade_eyes = "C:/Users/Chochstr/My Programs/opencv/data/haarcascades/haarcascade_eyes.xml";

string Window_name = "Detect Face(s) in image";
int im_width;
int im_height;

Ptr<FaceRecognizer> model;
Ptr<FaceRecognizer> model2;

string file_csv = "C:/Users/Chochstr/Pictures/classmates_faces/Myfileslist.txt";

// Vectors
vector<Mat> images;
vector<int> labels;

CvCapture* capture;
int prev_predict, predict_series;
Mat face_resized;

// Prediction Labels
int predicted_label;
int predicted_label1;

static void read_csv(const string& filename, vector<Mat>& images, vector<int>& labels, char separator = ';') {
    std::ifstream file(filename.c_str(), ifstream::in);
    if (!file) {
        string error_message = "No valid input file was given, please check the given filename.";
        CV_Error(CV_StsBadArg, error_message);
    }
    string line, path, classlabel;
    while (getline(file, line)) {
        stringstream liness(line);
		line.erase( 0, line.find_first_of("C"));
		path = line.substr(0, line.find(separator));
		classlabel = line.substr(line.find(separator)+1,line.length());
        if(!path.empty() && !classlabel.empty()) {
            images.push_back( imread(path, CV_LOAD_IMAGE_GRAYSCALE) );
            labels.push_back( atoi(classlabel.c_str()) );
        }
    }
}

void task1(void *param) {
	predicted_label = -1;
	double predicted_confidence1 = 0.0;
	model->predict(face_resized, predicted_label, predicted_confidence1);
	_endthread();
}

Mat DetectFace(Mat frame) {

	// Clone the frame image
	Mat original = frame.clone();
	Mat gray, face;
	string text;
	Rect face_i;

	// Gray out the image
	cvtColor(original, gray, CV_BGR2GRAY);
	equalizeHist(gray, gray);

	// Vector of Rectanglular faces
	vector<Rect> faces;
	bool check = true;

	// Prediction results
	double predicted_confidence1 = 0.0;
	double predicted_confidence = 0.0;
	string result, box_text;

	// Detect facess by gray scale
	haar.detectMultiScale(gray, faces, 1.3, 3, CV_HAAR_SCALE_IMAGE | CV_HAAR_DO_CANNY_PRUNING, Size(30,30));

	// Loop through all faces found by detection of image
	for(int i = 0; i < faces.size(); i++) {

		// Get a face
		face_i = faces[i];

		// Gray the face image
		face = gray(face_i);

		// Resize the image
		resize(face, face_resized, Size(im_width,im_height), 1.0, 1.0, INTER_CUBIC);
	
		// Fischer REcognizer
		HANDLE hThread;
		hThread = (HANDLE)_beginthread(task1, 0, NULL);

		// Eigen Recognizer
		predicted_label1 = -1;
		model2->predict(face_resized, predicted_label1, predicted_confidence1);

		// Join threads and only use one
		WaitForSingleObject(hThread, INFINITE);

		// Draw a rectangle
		rectangle(original, face_i, CV_RGB(255,255,255), 1);

		// Check if the Labels are the same
		if(predicted_label == predicted_label1)
			result = to_string(predicted_label1);

		// Get the result in
		box_text = result;

		// Position text
		int pos_x = max(face_i.tl().x - 10, 0);
		int pos_y = max(face_i.tl().y - 10, 0);

		// Add text with person(s) name
		putText(original, box_text, Point(pos_x,pos_y), FONT_HERSHEY_PLAIN, 1.0, CV_RGB(255,255,255), 1);
	}

	// Return the original image
	return original;
}

Mat Remap(Mat src) {
	Mat dst, map_x, map_y;
	dst.create(src.size(), CV_32FC1);
	map_x.create(src.size(), CV_32FC1);
	map_y.create(src.size(), CV_32FC1);
	for(int j = 0; j < src.rows; j++) {
		for( int i = 0; i < src.cols; i++) {
			map_x.at<float>(j,i) = src.cols - i;
			map_y.at<float>(j,i) = j;
		}
	}
	remap(src, dst, map_x, map_y, CV_INTER_LINEAR, BORDER_CONSTANT, Scalar(0,0,0));
	return dst;
}


void Image_Detect() {

	// Load Face Classifier
	haar.load(cascade_alt2);
	
	// Read CSV file and store images and labels
	try {
		read_csv(file_csv, images, labels);
	} catch (Exception& e) {
		cerr << "Error opening file \"" << file_csv << "\". Reason: " << e.msg << endl;
		exit(1);
	}
	im_width = images[0].rows;
	im_height = images[0].cols;

	// Creat FisheFace model for face recognizer and train it with
	// images and labels read from the given CSV file. FULL PCA
	model = createFisherFaceRecognizer(0, 123.0);
	model->train(images, labels);

	// Creat EigenFaces model for face recognizer and train it with
	// images and labels read from the given CSV file. FULL PCA
	model2 = createEigenFaceRecognizer(0, 123.0);
	model2->train(images, labels);

	// Set some stuff up
	prev_predict = -1;

	// Loop
	while(1){

		// Capture from Webcam
		capture = cvCaptureFromCAM(-1);

		// Make a frame
		Mat frame = cvQueryFrame(capture);

		// Detect face(s) in frame
		Mat dst = DetectFace(frame);

		// Create a window to show frame
		namedWindow(Window_name, CV_WINDOW_AUTOSIZE);

		// Remap image
		//dst = Remap(dst);

		// Show image
		imshow(Window_name, dst);

		// check if exit is pushed
		char c = waitKey(5);
		if (c >= 0) break;
	}
	cvReleaseCapture(&capture);
	destroyWindow(Window_name);
}