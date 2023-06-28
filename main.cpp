#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;


int main() {

	string image_path = "C:\\Users\\ryana\\OneDrive\\Desktop\\dog.jpg";

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

	// cout << image_pad.rows << "\t" << image_pad.cols << endl;

	// imshow("Image", image_pad);

	//openCV implementation of Gradient Calculation

	Mat x_grad = Mat(img_size, CV_32FC1);
	Mat y_grad = Mat(img_size, CV_32FC1);

	Sobel(image_pad, x_grad, CV_32FC1, 0, 1, 1);
	Sobel(image_pad, y_grad, CV_32FC1, 1, 0, 1);

	// Gives 2 matrices 
	Mat mag, dir;
	cartToPolar(x_grad, y_grad, mag, dir, true);

	// display images
	// imshow("X", x_grad);
	// imshow("Y", y_grad);
	// imshow("Magnitude", mag);

	std::cout << "\n\n" << size(mag) << "\n\n" << endl;

	// Determines how many histograms needed for each cell
	const int nBin_rows = (image_pad.rows / 8) * (image_pad.cols / 8);
	
	// Create 9x1 bins for each cell 
	double **HOGBin = new double*[nBin_rows];
	for(int i = 0; i < nBin_rows; ++i){
		HOGBin[i] = new double[9] {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
	}
	cout << "\n\n" << endl;

	for(int i = 0; i < nBin_rows; ++i){
		for(int j = 0; j < 9; ++j)
			cout << HOGBin[i][j] << " " << endl;
		cout << "\n" << endl;
	}
	

	cout << "\n\n" << endl;

	// Initialize bins


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

	// Free memory
	for (int i = 0; i < nBin_rows; ++i){
		delete[] HOGBin[i];
	}
	delete[] HOGBin;

	waitKey(0);

	return 0;
}

//----------Useful References -------------------
//https://learnopencv.com/histogram-of-oriented-gradients/
//https://docs.opencv.org/4.x/d6/d6d/tutorial_mat_the_basic_image_container.html
