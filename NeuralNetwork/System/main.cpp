#include <stdio.h>

#include "ArtificialNeuralNetwork.h"

void main() {

	//InputLayer* myinput = new InputLayer(2);
	//myinput->readDataSetFromCSV("Dataset/SinCosTan.csv", 3);
	
	{
		ArtificialNeuralNetwork* net = new ArtificialNeuralNetwork;
		net->addBasicLayer(LayerType::Input, 2);
		net->addBasicLayer(LayerType::FullyConnected, 2);
		net->addBasicLayer(LayerType::Sigmoid, 2);
		net->addBasicLayer(LayerType::FullyConnected, 2);
		net->addBasicLayer(LayerType::Sigmoid, 2);
		net->addBasicLayer(LayerType::Lost_MES, 2);

		net->setLearningRate(0.5f);
		net->setUpfortest();

		net->forwardPropagation();
		net->backPropagationOneByOne();
	}

	if (true) return;

	ArtificialNeuralNetwork* net = new ArtificialNeuralNetwork;
	net->addBasicLayer(LayerType::Input, 2);
	net->addBasicLayer(LayerType::FullyConnected, 5);
	net->addBasicLayer(LayerType::Sigmoid, 5);
	net->addBasicLayer(LayerType::FullyConnected, 6);
	net->addBasicLayer(LayerType::Sigmoid, 6);
	net->addBasicLayer(LayerType::Lost_MES, 6);


	net->getInputLayer()->readDataSetFromCSV("Dataset/SinCosTan.csv", 3);

	net->forwardPropagation();

	printf("aaa");
	getchar();
}