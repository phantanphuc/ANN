#ifndef __INPUTLAYER_H__
#define __INPUTLAYER_H__

#include "BaseLayer.h"
#include <stdio.h>

typedef float decimal;

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
	decimal* getDataSet();

private:

	char* rawDataBuffer;
	int buffer_data_len; // num of record / line
	/*
	The DataBuffer has structure:
	[
		input1, target1, input2, target2, ....
	]
	*/
	decimal * DataBuffer;
	int stride;

};
#endif
