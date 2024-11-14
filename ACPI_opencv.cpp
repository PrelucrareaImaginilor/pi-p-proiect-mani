// ACPI_opencv.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <iostream>
#include <chrono>
#include "operatii.h"
#include "SpineDetector.h"
using namespace std;
using namespace cv;

int main()
{
    ///*Citire imagine*/
    //Mat img = imread("Images/lena512.bmp", IMREAD_GRAYSCALE);

    ///*determinare latime, inaltime imagine*/
    //int w = img.size().width, h = img.size().height;

    ///*Negativare imagine*/
    //unsigned char* neg = negateImage(img.data, w, h);
    //Mat negatedImg(img.size(), img.type(), neg);


    ///*Afisare rezultate*/
    //imshow("Imagine originala", img);
    //imshow("Imagine inversa", negatedImg);
    //

    //waitKey();
    Mat img = imread("C:\\Users\\maria\\Desktop\\ACPI_opencv\\ACPI_opencv\\ACPI_opencv\\Images\\2.jpeg");
    //
    if (img.empty()) {
        cout << "Nu s-a putut încărca imaginea!" << endl;
        return -1;
    }
    auto start = chrono::high_resolution_clock::now();
    imshow("Imagine Originala", img);
    waitKey(0);
    Mat margine = SpineDetector::preprocesare((Mat)img);
    vector<vector<Point>> contur = SpineDetector::gasireContur(margine);
    //vector<vector<cv::Point>> filteredContours = SpineDetector::filterContours(contours);
    imshow("Imagine preprocesata", margine);
    waitKey(0);
    cout << "evaluare metrici:" << endl;
    evalueazaMetriciGeometrice(contur, img.size());

    cv::Mat rez;
    img.copyTo(rez);
    drawContours(rez, contur, -1, Scalar(255, 128, 0), 2);

    
    cv::imshow("Coloana Vertebrala Detectata", rez);
    cv::waitKey(0);
    auto end = chrono::high_resolution_clock::now();
    chrono::duration<double> durata = end - start;
    cout << "Timpul total de executie al procesului de segmentare si calculul ariei: "
        << durata.count() << " secunde" << endl;


    return 0;
}
