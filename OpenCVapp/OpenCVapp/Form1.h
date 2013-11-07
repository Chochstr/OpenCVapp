#pragma once

// Libraires to be included
#include <stdio.h>
#include "resource.h"
#include <string>
#include <fstream> 
#include <sstream>
#include <iostream> 
#include <direct.h>
#include <Windows.h>
#include <process.h>
#include "core.hpp"
#include "imgproc.hpp"
#include "highgui.hpp"
#include "contrib.hpp"
#include "objdetect.hpp"
#include "flann\miniflann.hpp"
#include "photo\photo.hpp" 
#include "video\video.hpp" 
#include "features2d\features2d.hpp" 
#include "calib3d\calib3d.hpp" 
#include "opencv2\opencv.hpp"
#include "ml\ml.hpp" 
#include <msclr\marshal.h>

namespace OpenCVapp {

	// Add namespace's
	using namespace System;
	using namespace System::ComponentModel;
	using namespace System::Collections;
	using namespace System::Windows::Forms;
	using namespace System::Data;
	using namespace System::Drawing;
	using namespace System::Runtime::InteropServices;
	using namespace cv;
	using namespace std;

	/// <summary>
	/// Summary for Form1
	/// </summary>
	public ref class Form1 : public System::Windows::Forms::Form
	{
	public:
		Form1(void)
		{
			InitializeComponent();
			//
			//TODO: Add the constructor code here
			//
		}

	protected:
		/// <summary>
		/// Clean up any resources being used.
		/// </summary>
		~Form1()
		{
			if (components)
			{
				delete components;
			}
		}

	protected: 
	private: System::Windows::Forms::Button^  button1;
	private: System::Windows::Forms::Button^  button2;
	private: System::Windows::Forms::TextBox^  textBox1;
	private: System::Windows::Forms::Label^  label1;
	private: System::Windows::Forms::Button^  button3;

	private:
		/// <summary>
		/// Required designer variable.
		/// </summary>
		System::ComponentModel::Container ^components;

#pragma region Windows Form Designer generated code
		/// <summary>
		/// Required method for Designer support - do not modify
		/// the contents of this method with the code editor.
		/// </summary>
		void InitializeComponent(void)
		{
			this->button1 = (gcnew System::Windows::Forms::Button());
			this->button2 = (gcnew System::Windows::Forms::Button());
			this->textBox1 = (gcnew System::Windows::Forms::TextBox());
			this->label1 = (gcnew System::Windows::Forms::Label());
			this->button3 = (gcnew System::Windows::Forms::Button());
			this->SuspendLayout();
			// 
			// button1
			// 
			this->button1->Location = System::Drawing::Point(12, 53);
			this->button1->Name = L"button1";
			this->button1->Size = System::Drawing::Size(75, 23);
			this->button1->TabIndex = 1;
			this->button1->Text = L"Build";
			this->button1->UseVisualStyleBackColor = true;
			this->button1->Click += gcnew System::EventHandler(this, &Form1::button1_Click);
			// 
			// button2
			// 
			this->button2->Location = System::Drawing::Point(188, 53);
			this->button2->Name = L"button2";
			this->button2->Size = System::Drawing::Size(75, 23);
			this->button2->TabIndex = 2;
			this->button2->Text = L"Stop";
			this->button2->UseVisualStyleBackColor = true;
			this->button2->Click += gcnew System::EventHandler(this, &Form1::button2_Click);
			// 
			// textBox1
			// 
			this->textBox1->Location = System::Drawing::Point(86, 12);
			this->textBox1->Name = L"textBox1";
			this->textBox1->Size = System::Drawing::Size(145, 20);
			this->textBox1->TabIndex = 3;
			this->textBox1->Text = L"Please Enter Name";
			this->textBox1->Enter += gcnew System::EventHandler(this, &Form1::textBox1_Enter);
			this->textBox1->Leave += gcnew System::EventHandler(this, &Form1::textBox1_Leave);
			// 
			// label1
			// 
			this->label1->AutoSize = true;
			this->label1->Location = System::Drawing::Point(28, 15);
			this->label1->Name = L"label1";
			this->label1->Size = System::Drawing::Size(38, 13);
			this->label1->TabIndex = 4;
			this->label1->Text = L"Name:";
			// 
			// button3
			// 
			this->button3->Location = System::Drawing::Point(101, 53);
			this->button3->Name = L"button3";
			this->button3->Size = System::Drawing::Size(75, 23);
			this->button3->TabIndex = 5;
			this->button3->Text = L"Start";
			this->button3->UseVisualStyleBackColor = true;
			this->button3->Click += gcnew System::EventHandler(this, &Form1::button3_Click);
			// 
			// Form1
			// 
			this->AutoScaleDimensions = System::Drawing::SizeF(6, 13);
			this->AutoScaleMode = System::Windows::Forms::AutoScaleMode::Font;
			this->ClientSize = System::Drawing::Size(275, 88);
			this->Controls->Add(this->button3);
			this->Controls->Add(this->label1);
			this->Controls->Add(this->textBox1);
			this->Controls->Add(this->button2);
			this->Controls->Add(this->button1);
			this->Cursor = System::Windows::Forms::Cursors::Default;
			this->Name = L"Form1";
			this->Text = L"OpenCVapp";
			this->FormClosing += gcnew System::Windows::Forms::FormClosingEventHandler(this, &Form1::Form1_FormClosing);
			this->Load += gcnew System::EventHandler(this, &Form1::Form1_Load);
			this->ResumeLayout(false);
			this->PerformLayout();

		}
#pragma endregion

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

	System::String ^TrimWhiteSpace(System::String ^str) {
		return System::Text::RegularExpressions::Regex::Replace(str, "^\\s+", System::String::Empty);
	}

	void MarshalString ( System::String ^ s, string& os ) {
		s = TrimWhiteSpace(s);
		const char* chars = (const char*)(Marshal::StringToHGlobalAnsi(s)).ToPointer();
		os = chars;
		Marshal::FreeHGlobal(IntPtr((void*)chars));
	}

	int SaveFace(Mat& frame, int count) {
		CascadeClassifier haar;
		haar.load("C:/Users/Chochstr/My Programs/opencv/data/haarcascades/haarcascade_frontalface_alt2.xml");
		Mat dst = frame.clone();
		Mat gray;
		cvtColor(dst, gray, CV_BGR2GRAY);
		equalizeHist(gray, gray);
		vector<Rect> faces;
		haar.detectMultiScale(gray, faces, 1.3, 3, CV_HAAR_FIND_BIGGEST_OBJECT | CV_HAAR_SCALE_IMAGE | CV_HAAR_DO_CANNY_PRUNING, cvSize(30,30));
		for(int i = 0; i < faces.size(); i++) {
			Rect face_i = faces[i];
			Mat face = gray(face_i);
			Mat face_resized;
			resize(face, face_resized, cvSize(92, 112), 1.0, 1.0, INTER_CUBIC);
			rectangle(frame, face_i, CV_RGB(255,255,255), 5);
			int pos_x = max(face_i.tl().x - 10, 0);
			int pos_y = max(face_i.tl().y - 10, 0);
			putText(frame, to_string(count), cvPoint(pos_x,pos_y), FONT_HERSHEY_PLAIN, 1.0, CV_RGB(255,255,255), 2.0);
			string person;
			System::String^ per = textBox1->Text;
			MarshalString(per,person);
			string folder = "C:/Users/Chochstr/Pictures/classmates_faces/" + person;
			mkdir(folder.c_str());
			string filename = folder + "/" + format("%d",count) + ".png";
			imwrite(filename, face_resized);
			count++;
		}
		return count;
	}

	// Global
	bool status;

	private: System::Void Form1_Load(System::Object^  sender, System::EventArgs^  e) {
				 status = false;
			 }

	private: System::Void button1_Click(System::Object^  sender, System::EventArgs^  e) {
				 string person;
				 System::String^ per = textBox1->Text;
				 MarshalString(per,person);
				 if (person != "" && person != "Please Enter Name") { 
					status = true;
					CvCapture* capture;
					capture = cvCaptureFromCAM(-1);
					int count = 0;
					while(count <= 100) {
						Mat frame  = cvQueryFrame(capture);
						frame = Remap(frame);
						count = SaveFace(frame, count);
						imshow("OpenCV Capture Window", frame);
						waitKey(5);
						if (status == false) break;
					}
					cvReleaseCapture(&capture);
					destroyWindow("OpenCV Capture Window");
					status = false;
				 }
			 }
	
	private: System::Void button2_Click(System::Object^  sender, System::EventArgs^  e) {
				 destroyWindow("OpenCV Capture Window");
				 status = false;
			 }
	
	private: System::Void Form1_FormClosing(System::Object^  sender, System::Windows::Forms::FormClosingEventArgs^  e) {
				 destroyAllWindows();
				 status = false;
			}

	private: System::Void textBox1_Enter(System::Object^  sender, System::EventArgs^  e) {
			 	 string person;
				 System::String^ per = textBox1->Text;
				 MarshalString(per,person);
				 if (person == "Please Enter Name") {
					 textBox1->Text = "";
				 }
			}

	private: System::Void textBox1_Leave(System::Object^  sender, System::EventArgs^  e) {
				 string person;
				 System::String^ per = textBox1->Text;
				 MarshalString(per,person);
				 if (person == "") {
					 textBox1->Text = "Please Enter Name";
				 }
			}

	private: System::Void button3_Click(System::Object^  sender, System::EventArgs^  e) {
				 Image_Detect();
			 }
};
}


