#include <stdio.h>

#include "ArtificialNeuralNetwork.h"

void main() {

	InputLayer* myinput = new InputLayer(2);
	myinput->readDataSetFromCSV("Dataset/SinCosTan.csv", 3);
	printf("aaa");
	getchar();
}