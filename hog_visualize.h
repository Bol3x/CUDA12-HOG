#ifndef HOG_VISUALIZE_H
#define HOG_VISUALIZE_H

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>

using namespace cv;

void visualizeHOG(const Mat& input_image, double* HOGFeatures, double*** HOGBin, int ncell_rows, int ncell_cols, int num_bins);

#endif // HOG_VISUALIZE_H