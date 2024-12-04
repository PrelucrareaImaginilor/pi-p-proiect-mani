#pragma once
#ifndef SPINEDETECTOR_H
#define SPINEDETECTOR_H

#include <opencv2/opencv.hpp>
#include <vector>

using namespace cv;
using namespace std;
class SpineDetector {
public:
    SpineDetector();
    static Mat preprocesare(const Mat& img);
    static vector<vector<Point>> gasireContur(const Mat& margine);
    static vector<vector<Point>> filtru(const vector<vector<Point>>& contur);
};

#endif