#include "InputLayer.h"


InputLayer::InputLayer(int layersize) : BaseLayer(layersize)
{
}

InputLayer::~InputLayer(){

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

	int DataBufferSize = stride * buffer_data_len;
	DataBuffer = new decimal[DataBufferSize];
	//decimal DataBuffer[1250]; // For debug
	decimal* writePtr = DataBuffer;

	
	for (int i = 0; i < DataBufferSize; ++i) {
		sscanf_s(ptr, "%f", writePtr);
		writePtr++;
		while ((*ptr != '\n') && (*ptr != ',')) ptr++; ptr++;
	}
}

decimal * InputLayer::getDataSet()
{
	return DataBuffer;
}
