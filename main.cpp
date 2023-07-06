#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <cmath>


using namespace cv;
using namespace std;


int main() {

	string image_path = "C:\\Users\\Carlo\\Downloads\\robot.png";

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

	// imshow("Image", image_pad);

	//openCV implementation of Gradient Calculation

	Mat x_grad = Mat(img_size, CV_32FC1);
	Mat y_grad = Mat(img_size, CV_32FC1);

	Sobel(image_pad, x_grad, CV_32FC1, 0, 1, 1);
	Sobel(image_pad, y_grad, CV_32FC1, 1, 0, 1);

	Mat mag, dir;
	cartToPolar(x_grad, y_grad, mag, dir, true);

	for (int i = 0; i < dir.rows; i++) {
		for (int j = 0; j < dir.cols; j++) {
			//bad modulo function impl (cuz its a float)
			if (dir.at<float>(i, j) > 180)
				dir.at<float>(i, j) = dir.at<float>(i, j) - 180;
		}
	}


	// display images
	// imshow("X", x_grad);
	// imshow("Y", y_grad);

	// Determines how many histograms needed for each cell
	const int ncell_rows = image_pad.rows / 8;
	const int ncell_cols = image_pad.cols / 8;
	const int nBin_size = ncell_rows * ncell_cols;
	
	// Create 9x1 bins for each cell in the bin matrix
	double ***HOGBin = new double**[ncell_rows];
	for(int i = 0; i < ncell_rows; ++i){
		HOGBin[i] = new double* [ncell_cols];
		for (int j = 0; j < ncell_cols; ++j) {
			HOGBin[i][j] = new double[9] {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
		}
	}

	
	/*
		Initialize the Bins
		Binning Method : Method 4 
		Source: https://www.analyticsvidhya.com/blog/2019/09/feature-engineering-images-introduction-hog-feature-descriptor/#h-step-4-calculate-histogram-of-gradients-in-8x8-cells-9x1

		* bins are are stored sequentially - index 0 = bin 0 and index 8 = bin 160
		* Magnitude of the corresponding pixel is distributed
	*/
	int bin_key = 0; 
	double angle = 0.0, bin_value_lo = 0.0, bin_value_hi = 0.0;
	const double bins[10] = { 0.0, 20.0, 40.0, 60.0, 80.0, 100.0, 120.0, 140.0, 160.0, 180.0 };

	int HOG_row = 0, HOG_col = 0;

// Traverse each cell in the original data
for(int i = 0; i < dir.rows; i++){
	for(int j = 0; j < dir.cols; j++){
	//get corresponding HOG bin block
	HOG_row = i/8;
	HOG_col = j/8;
	
	//angle 
	angle = dir.at<float>(i, j);

	// Round down to get which bin the direction belong to.
	bin_key = angle/20;
	bin_key %= 9;

	//special case for 180 - move value to 0 bin (bins wrap around)
	if (angle == 180.0) {
		angle = 0;
	}

	//equally divide contributions to different angle bins
	bin_value_lo = ((bins[bin_key+1] - angle)/ 20.0) * mag.at<float>(i, j);
	bin_value_hi = ((angle - bins[bin_key])/20.0) * mag.at<float>(i, j);

	//add value to bin
	HOGBin[HOG_row][HOG_col][bin_key] += bin_value_lo;
	HOGBin[HOG_row][HOG_col][(bin_key + 1) % 9] += bin_value_hi;
	}
}
	
	// Optional : Shows the HOG distribution of each cell 
	cout << "Result: " << endl;
	for (int i = 0; i < ncell_rows; ++i) {
		cout << "Row " << i << endl;
		for (int j = 0; j < ncell_cols; ++j) {
			for (int k = 0; k < 9; ++k) {
				cout << HOGBin[i][j][k] << "\t";
			}
			cout << endl;
		}

	}
	cout << endl;

	//todo: 2x2 blocking of histogram coefficients (into 1x36 coeffs)
	


	//todo: normalization (L2 Norm) of resulting gradients




	// cout << dir ;
	//todo: display HOG



	// Free memory
	for (int i = 0; i < ncell_rows; ++i){
		for(int j = 0; j < ncell_cols; ++j){
			delete[] HOGBin[i][j];
		}
		delete[] HOGBin[i];
	}
	delete[] HOGBin;
	
	waitKey(0);
	return 0;
}

//----------Useful References -------------------
//https://learnopencv.com/histogram-of-oriented-gradients/
//https://docs.opencv.org/4.x/d6/d6d/tutorial_mat_the_basic_image_container.html
//https://www.analyticsvidhya.com/blog/2019/09/feature-engineering-images-introduction-hog-feature-descriptor/#h-step-4-calculate-histogram-of-gradients-in-8x8-cells-9x1
