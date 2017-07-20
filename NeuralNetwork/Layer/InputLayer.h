#ifndef __INPUTLAYER_H__
#define __INPUTLAYER_H__

#include <iostream>
#include <fstream>
#include <vector>
#include "BaseLayer.h"
#include <stdio.h>

//#include "opencv2/core/core.hpp"
//#include "opencv2/imgproc/imgproc.hpp"
//#include "opencv2/highgui/highgui.hpp"

//using namespace cv;
using namespace std;

enum InputType {
	NORMAL, MNIST
};

class InputLayer : public BaseLayer {
public:
	InputLayer(int layersize);
	~InputLayer();

	/*
	This method load data from csv file
	path:			path of csv file
	targetsize:		size of output vector
	*/

	//void readDataSetFromMNIST(char* pathdata, char* parTarget);
	void readDataSetFromCSV(char* path, int targetsize);
	void setTestDataset();
	decimal* getDataSet();
	
	void forwardPropagation();

	int getDataStride() { return stride; };
	int getTargetSize() { return target_size; };

	void Next();
	void resetLayer();

	decimal* getZ();

	//void read_Mnist(string filename, vector<cv::Mat> &vec);

private:
	InputType inputtype = NORMAL;

	int reading_index;
	//vector<cv::Mat> image_vector;


	char* rawDataBuffer = nullptr;
	int buffer_data_len; // num of record / line
	/*
	The DataBuffer has structure:
	[
		input1, target1, input2, target2, ....
	]
	*/
	decimal * DataBuffer = nullptr;
	int stride;
	int target_size;

	decimal* bias_w;
};
#endif
