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
#include <vector>
#include <string>

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

// Window name
string Window_name = "Detect Face(s) in image";

// image width && height
int im_width;
int im_height;

// Pointer Face Recongnizer
Ptr<FaceRecognizer> Fisher_model;
Ptr<FaceRecognizer> Eigen_model;
Ptr<FaceRecognizer> LBPH_model;

// File of path names
string file_csv = "C:/Users/Chochstr/Pictures/classmates_faces/Myfileslist.txt";

// Store people into person struct
struct person {
	int num;
	string name;
} p;

// Vectors
vector<Mat> images;
vector<int> labels;
vector<person> people;

CvCapture* capture;
int prev_predict, predict_series;
Mat face_resized;

// Prediction Labels
int Fisher_Predict, Eigen_Predict, LBPH_Predict;

// Prediction Confidence
double Fisher_Confidence = 0.0, Eigen_Confidence = 0.0, LBPH_Confidence = 0.0;

// Read file of images and store for Recognition
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
			path.resize(path.find_last_of("/\\"));
			p.name = path.substr(path.find_last_of("/\\")+1);
			p.num = atoi(classlabel.c_str());
			people.emplace_back(p);
        }
    }
}

void task1(void *param) {
	Fisher_Predict = -1;
	Fisher_model->predict(face_resized, Fisher_Predict, Fisher_Confidence);
	_endthread();
}

void task2(void *param) {
	LBPH_Predict = -1;
	LBPH_model->predict(face_resized, LBPH_Predict, LBPH_Confidence);
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
	
		// Fischer Recognizer
		HANDLE hThread;
		hThread = (HANDLE)_beginthread(task1, 0, NULL);

		// LBPH Recognizer
		HANDLE hThread2;
		hThread2 = (HANDLE)_beginthread(task2, 0, NULL);

		// Eigen Recognizer
		Eigen_Predict = -1;
		Eigen_model->predict(face_resized, Eigen_Predict, Eigen_Confidence);

		// Join threads and only use one
		WaitForSingleObject(hThread, INFINITE);
		WaitForSingleObject(hThread2, INFINITE);

		// Draw a rectangle
		rectangle(original, face_i, CV_RGB(255,255,255), 1);

		// Check if the Labels are the same
		if (Fisher_Predict == Eigen_Predict && Fisher_Predict == LBPH_Predict) {

			if (LBPH_Predict != -1) {

				for (vector<person>::size_type i = 0; i != people.size(); i++) {

					if (LBPH_Predict == people[i].num) {

						// Get the name of the person
						result = people[i].name;

						// store previously found prediction
						prev_predict = LBPH_Predict;
						
						// Get the result in
						box_text = result;

						// Position text
						int pos_x = max(face_i.tl().x - 10, 0);
						int pos_y = max(face_i.tl().y - 10, 0);

						// Add text with person(s) name
						putText(original, box_text, Point(pos_x,pos_y), FONT_HERSHEY_PLAIN, 1.0, CV_RGB(255,255,255), 1);
					}
				}
			}
		}
	}

	// Return the original image
	return original;
}

// ReMap the image
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


// Creat FisheFace model for face recognizer and train it with
// images and labels read from the given CSV file. FULL PCA
void Model_Fisher(void *param) {
	Fisher_model = createFisherFaceRecognizer();
	Fisher_model->set("threshold",0.0);
	Fisher_model->train(images, labels);
	_endthread();
}

// Creat EigenFaces model for face recognizer and train it with
// images and labels read from the given CSV file. FULL PCA
void Model_Eigen(void *param) {
	Eigen_model = createEigenFaceRecognizer();
	Eigen_model->set("threshold",0.0);
	Eigen_model->train(images, labels);
	_endthread();
}

void Image_Detect() {

	// Change Cursor to wait for trainning
	System::Windows::Forms::Cursor::Current = System::Windows::Forms::Cursors::WaitCursor;

	// Load Face Classifier
	haar.load(cascade_alt2);
	
	// Read CSV file and store images and labels
	try {
		read_csv(file_csv, images, labels);
	} catch (Exception& e) {
		cerr << "Error opening file \"" << file_csv << "\". Reason: " << e.msg << endl;
		exit(1);
	}

	// Get the width and height of images
	im_width = images[0].rows;
	im_height = images[0].cols;

	// Create a thread to run FisherFace model
	HANDLE hThread;
	hThread = (HANDLE)_beginthread(Model_Fisher, 0, NULL);

	// Create a thread to run EigenFace model
	HANDLE hThread2;
	hThread2 = (HANDLE)_beginthread(Model_Eigen,0,NULL);

	// Creat LBPH model for face recognizer and train it with
	// images and labels read from the given CSV file. FULL PCA
	LBPH_model = createLBPHFaceRecognizer();
	LBPH_model->set("threshold",0.0);
	LBPH_model->train(images, labels);

	// Join threads and only use one
	WaitForSingleObject(hThread, INFINITE);
	WaitForSingleObject(hThread2, INFINITE);

	// Set some stuff up
	prev_predict = -1;

	// Set Cursor to default
	System::Windows::Forms::Cursor::Current = System::Windows::Forms::Cursors::Default;

	// Loop
	while(1) {

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

	// Release the image
	cvReleaseCapture(&capture);
	destroyWindow(Window_name);
}