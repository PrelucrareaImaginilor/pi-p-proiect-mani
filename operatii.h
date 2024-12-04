#pragma once
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <vector>
#include <iostream>
using namespace std;
#include <opencv2/opencv.hpp>
using namespace cv;

//unsigned char* negateImage(unsigned char* img, int w, int h)
//{
//	unsigned char* result = new unsigned char[w * h];
//	for (int y = 0; y < h; y++)
//		for (int x = 0; x < w; x++)
//			result[y * w + x] = 255 - img[y * w + x];
//	return result;
//}
void evalueazaMetriciGeometrice(const vector<vector<Point>>& contururi, const Size& dimensiuneImagine)
{
	double ariaTotala = 0;
	double aria;
	for (size_t i = 0; i < contururi.size(); i++) { 
		 aria= contourArea(contururi[i]);
		ariaTotala += aria;
		
		}
	cout << ariaTotala << endl;
	if (ariaTotala < 1000 || ariaTotala > 5000) {
	 cout << "Atentie!! Arie prea mare sau prea mica" << endl;
	}
}