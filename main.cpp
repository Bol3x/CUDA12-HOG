#define _USE_MATH_DEFINES

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <opencv2/opencv.hpp>

#include <cmath>

#include <time.h>


using namespace cv;
using namespace std;


void visualizeHOG(const Mat input_image,  double* HOGFeatures,  double ***HOGBin, int ncell_rows, int ncell_cols, int num_bins) {
    // Create a grayscale visualization image with the same size as the input image
    Mat hog_visualization = Mat(input_image.rows, input_image.cols, CV_8UC1, Scalar(0));

    // Calculate the size of each cell in the visualization
    int cell_width = input_image.cols / ncell_cols;
    int cell_height = input_image.rows / ncell_rows;

    int feature_idx = 0;
    for (int r = 0; r < ncell_rows; ++r) {
        for (int c = 0; c < ncell_cols; ++c) {
            // // Calculate the center position of the cell
			// since there is a new block on each cell (overlapping blocks) but the last one
            double center_x = (c + 1) * cell_width;
            double center_y = (r + 1) * cell_height;

            // Get the dominant direction stored in the HOGBin array
            double dominant_direction[4];
			dominant_direction[0] = 0.0;
			dominant_direction[1] = 0.0;
			dominant_direction[2] = 0.0;
			dominant_direction[3] = 0.0;

            int max_bin_idx[4];

            for (int bin_idx = 0; bin_idx < num_bins; bin_idx++) {
                if (HOGBin[r][c][bin_idx] > dominant_direction[0]) {
                    dominant_direction[2] = dominant_direction[3];
					max_bin_idx[3] = max_bin_idx[2];

					dominant_direction[1] = dominant_direction[2];
					max_bin_idx[2] = max_bin_idx[1];

					dominant_direction[1] = dominant_direction[0];
					max_bin_idx[1] = max_bin_idx[0];

					dominant_direction[0] = HOGBin[r][c][bin_idx];
					max_bin_idx[0] = bin_idx;
                }
            }

			for(int i = 0; i < 4; ++i){
				// Calculate the arrow endpoint position based on the dominant gradient direction

				// add
				float currRad = max_bin_idx[i] * 3.14 / num_bins;
                float dirVecX = cos(currRad);
                float dirVecY = sin(currRad);
                float scale = 0.015; // just a visual_imagealization scale,
                                          // to see the lines better

				// compute line coordinates
                float x1 = center_x - dirVecX * dominant_direction[i] * scale;
                float y1 = center_y - dirVecY * dominant_direction[i] * scale;
                float x2 = center_x + dirVecX * dominant_direction[i] * scale;
                float y2 = center_y + dirVecY * dominant_direction[i] * scale;

				// Draw the arrowed line to represent the dominant gradient direction
				line(hog_visualization,
								Point(x1, y1),
								Point(x2, y2),
								Scalar(255),
								1, cv::LINE_AA);
			}
            feature_idx++;
        }
    }

    // Show the input image with the HOG visualization overlay
    cv::imshow("HOG Visualization", hog_visualization);
    cv::waitKey(0);
}

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

	string image_path = "C:\\Users\\admin\\Desktop\\dataset-iiit-pet-master\\images\\shiba_inu_45.jpg";

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
