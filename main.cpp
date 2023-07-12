#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <cmath>


using namespace cv;
using namespace std;


/**
*	Computes for the x and y gradients of a given input matrix
*	Performs a 1D sobel filter with the top and left values negative, while bottom and right values positive
*	Stores corresponding gradient value in x_grad and y_grad matrices
*/
void compute_gradients(Mat x_grad, Mat y_grad, Mat input_data) {
	//initialize first row as all 0's
	for (int j = 0; j < input_data.cols; j++)
		x_grad.at<float>(0, j) = 0;

	//compute x gradients
	float top, bottom;
	for (int i = 1; i < input_data.rows - 1; i++) {
		for (int j = 0; j < input_data.cols; j++) {
			bottom = input_data.at<float>(i + 1, j);
			top = input_data.at<float>(i - 1, j);

			x_grad.at<float>(i, j) = bottom - top;
		}
	}

	//initialize first column as all 0's
	for (int i = 0; i < input_data.rows; i++)
		y_grad.at<float>(i, 0) = 0;

	//compute y gradients
	float left, right;
	for (int i = 0; i < input_data.rows; i++) {
		for (int j = 1; j < input_data.cols - 1; j++) {
			right = input_data.at<float>(i, j + 1);
			left = input_data.at<float>(i, j - 1);

			y_grad.at<float>(i, j) = right - left;
		}
	}
}


/*
*	Computes the polar representation of the x and y magnitudes of each element in the 2 matrices.
*	Returns the magnitude and direction of each element and stores them in mag and dir respectively
*/
void compute_polar(Mat mag, Mat dir, Mat x_mag, Mat y_mag) {
	float x_val, y_val, res_dir;
	double pi = atan(1) * 4;
	//compute magnitude and direction of cartesian coordinates x_val and y_val
	for (int i = 0; i < x_mag.rows; i++) {
		for (int j = 0; j < x_mag.cols; j++) {
			x_val = x_mag.at<float>(i, j);
			y_val = y_mag.at<float>(i, j);

			mag.at<float>(i, j) = sqrt(pow(x_val, 2) + pow(y_val, 2));
			res_dir = atan2(y_val, x_val) * 180/pi;
			
			//if resulting direction is < 0, negate by adding 180
			if (res_dir < 0) res_dir += 180;
			dir.at<float>(i, j) = res_dir;
		}
	}
}


void bin_gradients(double*** HOGBin, Mat mag, Mat dir) {
	/*
	Initialize the Bins
	Binning Method : Method 4
	Reference: https://www.analyticsvidhya.com/blog/2019/09/feature-engineering-images-introduction-hog-feature-descriptor/#h-step-4-calculate-histogram-of-gradients-in-8x8-cells-9x1

	* bins are are stored sequentially - index 0 = bin 0 and index 8 = bin 160
	* Magnitude of the corresponding pixel is distributed
*/
	int bin_key = 0;
	double mag_val = 0.0, angle = 0.0, bin_value_lo = 0.0, bin_value_hi = 0.0;
	const double bins[10] = { 0.0, 20.0, 40.0, 60.0, 80.0, 100.0, 120.0, 140.0, 160.0, 180.0 };

	int HOG_row = 0, HOG_col = 0;

	// Traverse each cell in the original data
	for (int i = 0; i < dir.rows; i++) {
		for (int j = 0; j < dir.cols; j++) {
			//get corresponding HOG bin block
			HOG_row = i / 8;
			HOG_col = j / 8;

			//angle 
			angle = dir.at<float>(i, j);
			//mag
			mag_val = mag.at<float>(i, j);

			// Round down to get which bin the direction belong to.
			bin_key = angle / 20;
			bin_key %= 9;

			//special case for 180 - move value to 0 bin (bins wrap around)
			if (angle == 180.0) {
				angle = 0;
			}

			//equally divide contributions to different angle bins
			bin_value_lo = ((bins[bin_key + 1] - angle) / 20.0) * mag_val;
			bin_value_hi = fabs(bin_value_lo - mag_val);

			//add value to bin
			HOGBin[HOG_row][HOG_col][bin_key] += bin_value_lo;
			HOGBin[HOG_row][HOG_col][(bin_key + 1) % 9] += bin_value_hi;
		}
	}
}


int main() {

	/************************************************************
	*					1. Reading image data
	*************************************************************/

	string image_path = "C:\\Users\\Carlo\\Downloads\\images\\shiba_inu_60.jpg";

	//greyscale for now, we can update later
	Mat image = imread(image_path, IMREAD_GRAYSCALE);
	short block_size = 8;

	//pad image to make it divisible by block_size
	Mat image_pad;
	copyMakeBorder(image, image_pad, 
		0, block_size - image.rows % block_size, 
		0, block_size - image.cols % block_size, 
		BORDER_CONSTANT, Scalar(0));

	image_pad.convertTo(image_pad, CV_32FC1);

	Size img_size = image_pad.size();

	cout << image_pad.rows << "\t" << image_pad.cols << "\n" << endl;

	
	/************************************************************
	*					2. Computing Gradients
	*************************************************************/

	Mat x_grad = Mat(img_size, CV_32FC1);
	Mat y_grad = Mat(img_size, CV_32FC1);

	compute_gradients(x_grad, y_grad, image_pad);



	/************************************************************
	*					3. Computing Polar values
	*************************************************************/

	Mat mag = Mat(img_size, CV_32FC1); 
	Mat dir = Mat(img_size, CV_32FC1);

	compute_polar(mag, dir, x_grad, y_grad);

	cout << "Magnitude" << endl;
	for (int i = 0; i < 8; i++) {
		for (int j = 0; j < 8; j++) {
			cout << mag.at<float>(i, j) << "\t";
		}
		cout << endl;
	}
	cout << endl;


	cout << "direction" << endl;
	for (int i = 0; i < 8; i++) {
		for (int j = 0; j < 8; j++) {
			cout << dir.at<float>(i, j) << "\t";
		}
		cout << endl;
	}
	cout << endl;

	/************************************************************
	*			4. Binning Gradients to Angle bins
	*************************************************************/

	// Determines how many blocks are needed for the image
	const int ncell_rows = image_pad.rows / 8;
	const int ncell_cols = image_pad.cols / 8;
	const int nBin_size = ncell_rows * ncell_cols;
	
	// initialize 9x1 bins corresponding to each block on the image
	double ***HOGBin = new double**[ncell_rows];
	for(int i = 0; i < ncell_rows; ++i){
		HOGBin[i] = new double* [ncell_cols];
		for (int j = 0; j < ncell_cols; ++j) {
			HOGBin[i][j] = new double[9] {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
		}
	}

	bin_gradients(HOGBin, mag, dir);

	cout << "Bins" << endl;
	for (int i = 0; i < 1; i++) {
		for (int j = 0; j < 1; j++) {
			for (int k = 0; k < 9; k++)
				cout << HOGBin[i][j][k] << "\t";
			cout << endl;
		}
		cout << endl;
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
