#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

void get_HOG_features(double* HOG_features, Mat img, double*** HOGBin, int rows, int cols) {

	const double bins[10] = { 0.0, 20.0, 40.0, 60.0, 80.0, 100.0, 120.0, 140.0, 160.0, 180.0 };

	/***********************************************
	*					COMPUTE BINS
	************************************************/
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

	/***********************************************
	*				NORMALIZE GRADIENTS
	************************************************/

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

				sum += (temp0 * temp0) + (temp1 * temp1) + (temp2 * temp2) + (temp3 * temp3);
			}

			#pragma unroll
			for (int k = 0; k < 36; k++) {
				HOG_features[feature_index + k] /= sqrt(sum + (1e-6 * 1e-6));
			}

			feature_index += 36;
		}
	}
}