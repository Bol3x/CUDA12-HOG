#define _USE_MATH_DEFINES

#include "hog_visualize.h"
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <opencv2/opencv.hpp>

#include <cmath>

#include <time.h>


using namespace cv;
using namespace std;


void get_HOG_features(double* HOG_features, Mat img, double*** HOGBin, int rows, int cols) {

	const double bins[10] = { 0.0, 20.0, 40.0, 60.0, 80.0, 100.0, 120.0, 140.0, 160.0, 180.0 };

	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			double x = (i == 0 || i == rows - 1) ? 0 : img.at<unsigned char>(i + 1, j) - img.at<unsigned char>(i - 1, j);
			double y = (j == 0 || j == cols - 1) ? 0 : img.at<unsigned char>(i, j + 1) - img.at<unsigned char>(i, j - 1);

			double mag = sqrt(x * x + y * y);
			double dir = atan2(y, x) * 180 / M_PI;
			if (dir < 0) dir += 180;
			if (dir == 180) dir = 0;

			int HOG_row = i / 8;
			int HOG_col = j / 8;

			int bin_key = dir / 20.0;
			bin_key %= 9;

			double bin_value_lo = ((bins[bin_key + 1] - dir) / 20.0) * mag;
			double bin_value_hi = fabs(bin_value_lo - mag);

			HOGBin[HOG_row][HOG_col][bin_key] += bin_value_lo;
			HOGBin[HOG_row][HOG_col][(bin_key + 1) % 9] += bin_value_hi;
		}
	}

	/************************************************************
	*			4. Normalize Gradients in a 16x16 cell
	*************************************************************/

	const int hog_rows = rows / 8;
	const int hog_cols = cols / 8;
	
	const int features = (hog_rows - 1) * (hog_cols - 1) * 36;

	int feature_index = 0;
	for (int i = 0; i < hog_rows - 1; i++) {
		for (int j = 0; j < hog_cols - 1; j++) {
			double sum = 0;
			#pragma unroll
			for (int k = 0; k < 9; k++) {
				double temp0, temp1, temp2, temp3;
				temp0 = HOGBin[i][j][k];
				temp1 = HOGBin[i][j + 1][k];
				temp2 = HOGBin[i + 1][j][k];
				temp3 = HOGBin[i + 1][j + 1][k];
				HOG_features[feature_index + k] = temp0;
				HOG_features[feature_index + 9 + k] = temp1;
				HOG_features[feature_index + 18 + k] = temp2;
				HOG_features[feature_index + 27 + k] = temp3;

				sum += (temp0*temp0) + (temp1*temp1) + (temp2*temp2) + (temp3*temp3);
			}
			
			#pragma unroll
			for (int k = 0; k < 36; k++) {
				HOG_features[feature_index + k] /= sqrt(sum + (1e-6*1e-6));
			}

			feature_index += 36;
		}
	}
}

int main() {

	clock_t start, end;
	double time_elapsed = 0;
	int runs = 100;

	/************************************************************
	*					1. Reading image data
	*************************************************************/

	string image_path = "D:/Github_Repositories/CUDA12-HOG/input_img/dog.jpg";

	//greyscale for now, we can update later
	Mat image = imread(image_path, IMREAD_GRAYSCALE);
	short block_size = 8;

	resize(image,
		image,
		Size(image.cols - (image.cols % block_size),
			 image.rows - (image.rows % block_size))
		);
	imshow("Image", image);
	Size img_size = image.size();

	// cout << image_pad.rows << "\t" << image_pad.cols << "\n" << endl;

	const int ncell_rows = img_size.height / 8;
	const int ncell_cols = img_size.width / 8;;

	// initialize 9x1 bins corresponding to each block on the image
	double ***HOGBin = new double**[ncell_rows];
	for(int i = 0; i < ncell_rows; ++i){
		HOGBin[i] = new double* [ncell_cols];
		for (int j = 0; j < ncell_cols; ++j) {
			HOGBin[i][j] = new double[9] {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
		}
	}
	const int num_features = (ncell_rows - 1) * (ncell_cols - 1) * 36;
	double* HOG_features = (double*)malloc(sizeof(double) * num_features);

	for (int i = 0; i < runs; i++) {

		//reset HOGBin
		for (int n = 0; n < ncell_rows; n++) {
			for (int m = 0; m < ncell_cols; m++) {
				for (int l = 0; l < 9; l++) {
					HOGBin[n][m][l] = 0;
				}
			}
		}
		

		start = clock();
		get_HOG_features(HOG_features, image, HOGBin, img_size.height, img_size.width);
		end = clock();

		time_elapsed += ((double)(end - start));
	}

	time_elapsed = (time_elapsed * 1e6 / CLOCKS_PER_SEC) / runs;

	cout << "Average time elapsed (in us): " << time_elapsed << endl;

	cout << "Sample HOG norm (First 36 elements)" << endl;
	for (int i = 0; i < 36; i++) {
		cout << HOG_features[i] << "\t";
	}
	cout << endl;

	visualizeHOG(image, HOG_features, HOGBin, ncell_rows, ncell_cols,9);

	// Free memory
	for (int i = 0; i < ncell_rows; ++i) {
		for (int j = 0; j < ncell_cols; ++j) {
			delete[] HOGBin[i][j];
		}
		delete[] HOGBin[i];
	}
	delete[] HOGBin;

	free(HOG_features);


	return 0;
}

//----------Useful References -------------------
//https://learnopencv.com/histogram-of-oriented-gradients/
//https://docs.opencv.org/4.x/d6/d6d/tutorial_mat_the_basic_image_container.html
//https://www.analyticsvidhya.com/blog/2019/09/feature-engineering-images-introduction-hog-feature-descriptor/#h-step-4-calculate-histogram-of-gradients-in-8x8-cells-9x1
