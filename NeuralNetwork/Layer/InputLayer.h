#ifndef __INPUTLAYER_H__
#define __INPUTLAYER_H__

#include "BaseLayer.h"
#include <stdio.h>



class InputLayer : public BaseLayer {
public:
	InputLayer(int layersize);
	~InputLayer();

	/*
	This method load data from csv file
	path:			path of csv file
	targetsize:		size of output vector
	*/

	void readDataSetFromCSV(char* path, int targetsize);
	void setTestDataset();
	decimal* getDataSet();
	
	void forwardPropagation();

	int getDataStride() { return stride; };
	int getTargetSize() { return target_size; };

	void Next();
	void resetLayer();
private:

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
