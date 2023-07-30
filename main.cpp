#define _USE_MATH_DEFINES

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <cmath>
#include <time.h>

//test HOG
#include <opencv2/objdetect.hpp>


using namespace cv;
using namespace std;


void displayBlock(Mat img) {
	for (int i = 0; i < 8; i++) {
		for (int j = 0; j < 8; j++) {
			cout << img.at<float>(i, j) << "\t";
		}
		cout << endl;
	}
	cout << endl;
}


/**
*	Computes for the x and y gradients of a given input matrix
*	Performs a 1D sobel filter with the top and left values negative, while bottom and right values positive
*	Then computes the polar representation of the x and y magnitudes of each element in the 2 matrices
*	Returns the magnitude and direction of each element and stores them in mag and dir respectively
*/
void compute_gradients(Mat& mag, Mat& dir, Mat input_data, int rows, int cols) {

	float temp;
	int x, y;
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			x = (i == 0 || i == rows - 1) ? 0 : input_data.at<unsigned char>(i+1, j) - input_data.at<unsigned char>(i-1, j);
			y = (j == 0 || j == cols - 1) ? 0 : input_data.at<unsigned char>(i, j+1) - input_data.at<unsigned char>(i, j-1);

			mag.at<float>(i, j) = sqrt(x * x + y * y);
			temp = atan2(y, x) * 180 / M_PI;
			if (temp < 0) temp += 180;
			dir.at<float>(i, j) = temp;
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

void L2Normalization(double *HOGFeatures, int n){
	double norm;
	double temp;
	//loop over each 2x2 block
	for (int i = 0; i < n; i+=36) {
		norm = 0;
		//
		for (int k = 0; k < 36; k++) {
			temp = HOGFeatures[i + k];
			norm += temp * temp;
		}

		norm = sqrt(norm);

		for (int k = 0; k < 36; k++)
			HOGFeatures[i + k] /= sqrt(norm*norm + 1e-6*1e-6);
	}
}

void normalizeGradients(double *HOGFeatures, double ***HOGBin, int rows, int cols, int num_elem){
	int feature_index = 0;
	//copy-in bin data
	for (int n = 0; n < rows-1; n++) {
		for (int m = 0; m < cols-1; m++) {

			for (int i = 0; i < 9; i++) {
				HOGFeatures[feature_index + i] = HOGBin[n][m][i];
				HOGFeatures[feature_index + 9 + i] = HOGBin[n][m+1][i];
				HOGFeatures[feature_index + 18 + i] = HOGBin[n+1][m][i];
				HOGFeatures[feature_index + 27 + i] = HOGBin[n+1][m+1][i];
			}

			feature_index += 36;
		}
	}

	//perform L2 Norm on each 1x36 feature
	L2Normalization(HOGFeatures, num_elem);
}


int main() {

	clock_t start, end;
	double time_elapsed_grad, time_elapsed_bin, time_elapsed_norm;
	int runs = 100;

	/************************************************************
	*					1. Reading image data
	*************************************************************/

	string image_path = "C:\\Users\\Carlo\\Downloads\\images\\shiba_inu_60.jpg";

	//greyscale for now, we can update later
	Mat image = imread(image_path, IMREAD_GRAYSCALE);
	short block_size = 8;

	//resize image to be divisible by block_size
	resize(image,
		image,
		Size(image.cols - (image.cols % block_size),
			image.rows - (image.rows % block_size))
	);

	Size img_size = image.size();

	cout << "Image dimensions" << endl;
	cout << image.rows << "\t" << image.cols << "\n" << endl;


	/************************************************************
	*					2-3. Computing Polar values
	*************************************************************/

	Mat mag = Mat(img_size, CV_32FC1); 
	Mat dir = Mat(img_size, CV_32FC1);

	start = clock();

	for (int i = 0; i < runs; i++)
		compute_gradients(mag, dir, image, image.rows, image.cols);

	end = clock();

	time_elapsed_grad = ((double)(end - start)) * 1e6 / CLOCKS_PER_SEC / runs;

	cout << "Average time elapsed (in us): " << time_elapsed_grad << endl;

	/************************************************************
	*			4. Binning Gradients to Angle bins
	*************************************************************/

	// Determines how many blocks are needed for the image
	const int ncell_rows = img_size.height / 8;
	const int ncell_cols = img_size.width / 8;
	const int nBin_size = ncell_rows * ncell_cols;
	
	// initialize 9x1 bins corresponding to each block on the image
	double ***HOGBin = new double**[ncell_rows];
	for(int i = 0; i < ncell_rows; ++i){
		HOGBin[i] = new double* [ncell_cols];
		for (int j = 0; j < ncell_cols; ++j) {
			HOGBin[i][j] = new double[9] {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
		}
	}

	start = clock();
	
	for (int i = 0; i < runs; i++)
		bin_gradients(HOGBin, mag, dir);
	
	end = clock();

	time_elapsed_bin = ((double)(end - start)) * 1e6 / CLOCKS_PER_SEC / runs;

	cout << "Average time elapsed (in us): " << time_elapsed_bin << endl;

	/************************************************************
	*			5. Normalize Gradients in a 16x16 cell
	*************************************************************/

	// Determine number of features
	const int features = (ncell_rows-1) * (ncell_cols-1) * 36;
	double *HOGFeatures = new double[features];

	start = clock();

	for (int i = 0; i < runs; i++)
		normalizeGradients(HOGFeatures, HOGBin, ncell_rows, ncell_cols, features);

	end = clock();

	time_elapsed_norm = ((double)(end - start)) * 1e6 / CLOCKS_PER_SEC / runs;

	cout << "Average time elapsed (in us): " << time_elapsed_norm << endl;
	
	/************************************************************
	*			6.		Visualizing HOG Features
	*************************************************************/





	cout << "Total Average time for computation (in us): " << time_elapsed_grad + time_elapsed_bin + time_elapsed_norm << endl;

	// Free memory used for calculating HOG features
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
