#include "InputLayer.h"


InputLayer::InputLayer(int layersize) : BaseLayer(layersize)
{
	layer_type = LayerType::Input;
	Z_vector = nullptr;
}

InputLayer::~InputLayer(){
	if (rawDataBuffer) delete rawDataBuffer;
	if (DataBuffer) delete DataBuffer;
}

void InputLayer::readDataSetFromCSV(char * path, int targetsize)
{
	FILE* csvfile;
	fopen_s(&csvfile, path, "r");

	fseek(csvfile, 0, SEEK_END);
	int filesize = ftell(csvfile);
	fseek(csvfile, 0, SEEK_SET);

	rawDataBuffer = new char[filesize + 1];
	fread_s(rawDataBuffer, filesize + 1, 1, filesize, csvfile);
	////////////////////////////////////////////////////
	//////////////// GET NUM OF RECORD /////////////////
	////////////////////////////////////////////////////
	buffer_data_len = 0;
	char* ptr = rawDataBuffer;
	while (*ptr != '\0')
	{
		if (*ptr == '\n')
			buffer_data_len++;
		ptr++;
	}
	ptr = rawDataBuffer;

	stride = targetsize + layer_size;
	target_size = targetsize;

	int DataBufferSize = stride * buffer_data_len;
	DataBuffer = new decimal[DataBufferSize];
	//decimal DataBuffer[1250]; // For debug
	decimal* writePtr = DataBuffer;

	
	for (int i = 0; i < DataBufferSize; ++i) {
		sscanf_s(ptr, "%f", writePtr);
		writePtr++;
		while ((*ptr != '\n') && (*ptr != ',')) ptr++; ptr++;
	}

	Z_vector = DataBuffer;

}

void InputLayer::setTestDataset()
{
	DataBuffer = new decimal[4];

	stride = 4;
	target_size = 2;

	DataBuffer[0] = 0.05;
	DataBuffer[1] = 0.1;
	DataBuffer[2] = 0.01;
	DataBuffer[3] = 0.99;

	Z_vector = DataBuffer;
}

decimal * InputLayer::getDataSet()
{
	return DataBuffer;
}


void InputLayer::forwardPropagation()
{
	//if (Z_vector == nullptr) Z_vector = DataBuffer;
	//else Z_vector += stride;
}

void InputLayer::Next()
{
	Z_vector += stride;
}

void InputLayer::resetLayer()
{
	Z_vector = DataBuffer;
}
