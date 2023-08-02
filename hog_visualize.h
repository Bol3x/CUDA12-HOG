#ifndef HOG_VISUALIZE_H
#define HOG_VISUALIZE_H

#include <opencv2/core/core.hpp>

void visualizeHOG(const cv::Mat& input_image, double* HOGFeatures, double*** HOGBin, int ncell_rows, int ncell_cols, int num_bins);

#endif 