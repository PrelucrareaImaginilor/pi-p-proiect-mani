#include "SpineDetector.h"
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

Mat SpineDetector::preprocesare(const Mat& img)
{
	Mat gri, blurat, margini;
	cvtColor(img, gri, COLOR_BGR2GRAY);
	GaussianBlur(gri, blurat, Size(5, 5), 1.5);
	Canny(blurat, margini, 50, 150);
	return margini;
}

vector<vector<Point>> SpineDetector::gasireContur(const Mat& margini)
{
	vector<vector<Point>> contur;
	findContours(margini, contur, RETR_TREE, CHAIN_APPROX_SIMPLE);
	return contur;
}
vector<vector<Point>> SpineDetector::filtru(const vector<vector<Point>>& contur)
{
	return contur;
}