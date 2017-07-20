#include "InputLayer.h"


InputLayer::InputLayer(int layersize) : BaseLayer(layersize)
{
	layer_type = LayerType::Input;
	Z_vector = nullptr;
}

InputLayer::~InputLayer(){
	//if (rawDataBuffer) delete rawDataBuffer;
	//if (DataBuffer) delete DataBuffer;
}

int ReverseInt(int i)
{
	unsigned char ch1, ch2, ch3, ch4;
	ch1 = i & 255;
	ch2 = (i >> 8) & 255;
	ch3 = (i >> 16) & 255;
	ch4 = (i >> 24) & 255;
	return((int)ch1 << 24) + ((int)ch2 << 16) + ((int)ch3 << 8) + ch4;
}
/*
void InputLayer::read_Mnist(string filename, vector<cv::Mat> &vec) {
	ifstream file(filename, ios::binary);
	if (file.is_open())
	{
		int magic_number = 0;
		int number_of_images = 0;
		int n_rows = 0;
		int n_cols = 0;
		file.read((char*)&magic_number, sizeof(magic_number));
		magic_number = ReverseInt(magic_number);
		file.read((char*)&number_of_images, sizeof(number_of_images));
		number_of_images = ReverseInt(number_of_images);
		file.read((char*)&n_rows, sizeof(n_rows));
		n_rows = ReverseInt(n_rows);
		file.read((char*)&n_cols, sizeof(n_cols));
		n_cols = ReverseInt(n_cols);

		number_of_images = 4;

		for (int i = 0; i < number_of_images; ++i)
		{
			cv::Mat tp = Mat::zeros(n_rows, n_cols, CV_64FC1);
			for (int r = 0; r < n_rows; ++r)
			{
				for (int c = 0; c < n_cols; ++c)
				{
					unsigned char temp = 0;
					file.read((char*)&temp, sizeof(temp));

					

					tp.at<decimal>(r, c) = (float)temp / 255.0;
				}
			}
			vec.push_back(tp);

			

			

		}
	}
}





void InputLayer::readDataSetFromMNIST(char * pathdata, char * parTarget)
{
	inputtype = MNIST;
	read_Mnist(pathdata, image_vector);
	for (int i = 0; i < 28;++i) {
		for (int j = 0; j < 28;++j) {
			if (image_vector[3].at<decimal>(i, j)) {
				//printf("X");
				printf("%f", image_vector[3].at<decimal>(i, j));
			}
			else {
				printf(" ");
			}
		}
		printf("\n");
	}
	
	imshow("1st", image_vector[0]);
	waitKey();
	int m = 0;
	m++;

}
*/
void InputLayer::readDataSetFromCSV(char * path, int targetsize)
{
	inputtype = NORMAL;

	FILE* csvfile;
	fopen_s(&csvfile, path, "r");

	fseek(csvfile, 0, SEEK_END);
	int filesize = ftell(csvfile);
	fseek(csvfile, 0, SEEK_SET);

	rawDataBuffer = new char[filesize + 1];
	fread_s(rawDataBuffer, filesize, 1, filesize, csvfile);
	
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
		float data = 0;
		sscanf_s(ptr, "%f", &data);
		*writePtr = decimal(data);
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
	switch (inputtype)
	{
	case NORMAL:
		Z_vector += stride;
		break;
	case MNIST:
		reading_index++;
		break;
	default:
		break;
	}
	
}

void InputLayer::resetLayer()
{
	switch (inputtype)
	{
	case NORMAL:
		Z_vector = DataBuffer;
		break;
	case MNIST:
		break;
	default:
		break;
	}
	
	reading_index = 0;
}

decimal * InputLayer::getZ()
{

	switch (inputtype)
	{
	case NORMAL:
		return Z_vector;
	case MNIST:
		//return image_vector.at(reading_index).data();
		break;
	}

	return nullptr;
}
