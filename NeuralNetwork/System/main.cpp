#include <stdio.h>

#include "ArtificialNeuralNetwork.h"
enum Work {
	WMNIST, WXORR, WSIN
} work;

void main() {
	work = WSIN;
	
	switch (work)
	{
	case WMNIST:
	{
		ArtificialNeuralNetwork net = ArtificialNeuralNetwork();
		net.addBasicLayer(LayerType::Input, 2);
		net.addBasicLayer(LayerType::FullyConnected, 10);
		net.addBasicLayer(LayerType::Sigmoid, 10);
		net.addBasicLayer(LayerType::FullyConnected, 1);
		net.addBasicLayer(LayerType::Sigmoid, 1);
		net.addBasicLayer(LayerType::Lost_MES, 1);

		//net.getInputLayer()->readDataSetFromMNIST("Dataset/MNIST/train-images.idx3-ubyte", "");
	}
		break;
	case WXORR:
	{
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
	}
		break;
	case WSIN:
	{
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
		net->setLearningRate(00.02f);
		decimal loss_nontrain = net->simpleEvaluateAll(datasetsize);

		for (int i = 0; i < 500; ++i) {
			net->trainOneByOne(datasetsize);
		}


		net->printAllWeight();
		decimal loss_train = net->simpleEvaluateAll(datasetsize);
		net->writeDownData("result.txt", 2000);

		printf("Loss Non  Train = %lf\n", loss_nontrain);
		printf("Loss with Train = %lf\n", loss_train);
	}
		break;
	default:
		break;
	}
	
	getchar();
}