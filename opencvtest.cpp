#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

  #include <windows.h>


int main() {
	// 
	char buffer[MAX_PATH];
  	::GetCurrentDirectory(MAX_PATH, buffer);
  	cout << "Current directory: " << buffer << endl;
	string image_path = ".\\input_img\\dog.jpg";

	//greyscale for now, we can update later
	Mat image = imread(image_path, IMREAD_GRAYSCALE);
	short block_size = 8;

	//pad image to make it divisible by block_size
	Mat image_pad;
	copyMakeBorder(image, image_pad, 
		0, block_size - image.rows % block_size, 
		0, block_size - image.cols % block_size, 
		BORDER_CONSTANT, Scalar(0));

	Size img_size = image_pad.size();

	cout << image_pad.rows << "\t" << image_pad.cols << endl;

	imshow("Image", image_pad);

	//openCV implementation of Gradient Calculation

	Mat x_grad = Mat(img_size, CV_32FC1);
	Mat y_grad = Mat(img_size, CV_32FC1);

	Sobel(image_pad, x_grad, CV_32FC1, 0, 1, 1);
	Sobel(image_pad, y_grad, CV_32FC1, 1, 0, 1);
	
	Mat x_grad_abs = cv::abs(x_grad);
	Mat y_grad_abs = cv::abs(y_grad);

	Mat mag, dir;
	cartToPolar(x_grad_abs, y_grad_abs, mag, dir, true);

	//debug: display images
	//imshow("X", x_grad);
	//imshow("Y", y_grad);
	//imshow("Magnitude", mag);

	cout << dir << endl;

	// int i, j;

	//loop through each block in the matrix (as per Dalal & Triggs' algorithm)
	// for (i = 0; i < image_pad.rows; i += block_size) {
	// 	for (j = 0; j < image_pad.cols; j += block_size) {

	// 		//todo: binning of gradients to angles


	// 	}
	// }


	//todo: 2x2 blocking of histogram coefficients (into 1x36 coeffs)
	


	//todo: normalization (L2 Norm) of resulting gradients


	
	//todo: display HOG


	waitKey(0);

	return 0;
}

//----------Useful References -------------------
//https://learnopencv.com/histogram-of-oriented-gradients/
//https://docs.opencv.org/4.x/d6/d6d/tutorial_mat_the_basic_image_container.html
