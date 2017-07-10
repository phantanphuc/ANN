#include <stdio.h>

#include "ArtificialNeuralNetwork.h"

void main() {

	if (false){
		ArtificialNeuralNetwork* net = new ArtificialNeuralNetwork;
		net->addBasicLayer(LayerType::Input, 2);
		net->addBasicLayer(LayerType::FullyConnected, 2);
		net->addBasicLayer(LayerType::Sigmoid, 2);
		net->addBasicLayer(LayerType::FullyConnected, 2);
		net->addBasicLayer(LayerType::Sigmoid, 2);
		net->addBasicLayer(LayerType::Lost_MES, 2);

		net->setLearningRate(0.2f);
		net->setUpfortest();

		
		net->forwardPropagation();
		printf("Loss Non  Train = %f\n", net->getLastLost());
		

		for(int c = 0; c < 5000; ++c) {
			net->forwardPropagation();
			net->backPropagationOneByOne();
		}
		
		net->forwardPropagation();
		net->printAllWeight();
		printf("Loss with Train = %f\n", net->getLastLost());
		getchar();
		if (true) return;
	}

	if (false) {
		ArtificialNeuralNetwork net = ArtificialNeuralNetwork();
		net.addBasicLayer(LayerType::Input, 2);
		net.addBasicLayer(LayerType::FullyConnected, 10);
		net.addBasicLayer(LayerType::Sigmoid, 10);
		net.addBasicLayer(LayerType::FullyConnected, 1);
		net.addBasicLayer(LayerType::Sigmoid, 1);
		net.addBasicLayer(LayerType::Lost_MES, 1);


		net.getInputLayer()->readDataSetFromCSV("Dataset/xor.csv", 1);
		int datasetsize = 4;
		net.linkLostLayer();

		//net->printAllWeight();
		decimal loss_nontrain = net.simpleEvaluateAll(datasetsize);
		net.setLearningRate(10.0f);
		for (int i = 0; i < 10000; ++i) {
			net.trainOneByOne(datasetsize);
		}

		//net->printAllWeight();
		decimal loss_train = net.simpleEvaluateAll(datasetsize);

		printf("Loss Non  Train = %f\n", loss_nontrain);
		printf("Loss with Train = %f\n", loss_train);
		getchar();
		return;
	}

	ArtificialNeuralNetwork* net = new ArtificialNeuralNetwork;
	net->addBasicLayer(LayerType::Input, 1);
	net->addBasicLayer(LayerType::FullyConnected, 25);
	net->addBasicLayer(LayerType::Sigmoid, 25);
	net->addBasicLayer(LayerType::FullyConnected, 1);
	net->addBasicLayer(LayerType::None, 1);
	net->addBasicLayer(LayerType::Lost_MES, 1);


	net->getInputLayer()->readDataSetFromCSV("Dataset/sin_new.csv", 1);
	net->linkLostLayer();


	net->printAllWeight();

	int datasetsize = 2000;
	net->setLearningRate(01.0f);
	decimal loss_nontrain = net->simpleEvaluateAll(datasetsize);
	
	for (int i = 0; i <	500; ++i) {
		net->trainOneByOne(datasetsize);
	}


	net->printAllWeight();
	decimal loss_train = net->simpleEvaluateAll(datasetsize);
	net->writeDownData("result.txt", 2000);
		
	printf("Loss Non  Train = %lf\n", loss_nontrain);
	printf("Loss with Train = %lf\n", loss_train);
	getchar();
}