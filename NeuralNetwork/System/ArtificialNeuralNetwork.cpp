#include "ArtificialNeuralNetwork.h"

#include "Layer\LostLayer_MSE.h"

ArtificialNeuralNetwork::ArtificialNeuralNetwork(){
	paramManager = new ParamManager;
}

ArtificialNeuralNetwork::~ArtificialNeuralNetwork(){
	delete paramManager;
}

void ArtificialNeuralNetwork::addLayer(BaseLayer * layer)
{
	if (!layer) return;
	layer->setParamManagerRef(paramManager);

	LayerNode* newnode = new LayerNode;
	newnode->layer = layer;
	newnode->nextNode = nullptr;

	if (head == nullptr) {
		head = newnode;
		tail = newnode;
	}
	else {
		for (LayerNode* ptr = head;; ptr = ptr->nextNode) {
			if (ptr->nextNode == nullptr) {
				ptr->nextNode = newnode;
				newnode->layer->setPreviousLayer(tail->layer);
				newnode->previousNode = ptr;
				tail = newnode;
				break;
			}
		}
	}
}

void ArtificialNeuralNetwork::addBasicLayer(LayerType layertype, int num_of_node)
{
	if (layertype == LayerType::Input) {
		InputLayer* inputlayer = new InputLayer(num_of_node);
		addLayer(inputlayer);
	}
	else if (layertype == LayerType::FullyConnected) {
		FullyConnectedLayer* fullyconnectedlayer 
			= new FullyConnectedLayer(num_of_node, tail->layer->getLayerSize());
		addLayer(fullyconnectedlayer);
	}
	else if (layertype == LayerType::Sigmoid) {
		ActivationLayer* sigmoidlayer = new ActivationLayer(num_of_node);
		sigmoidlayer->setActivationFunction(ActivationFunction::ACT_SIGMOID);
		addLayer(sigmoidlayer);
	}
	else if (layertype == LayerType::Tanh) {
		ActivationLayer* sigmoidlayer = new ActivationLayer(num_of_node);
		sigmoidlayer->setActivationFunction(ActivationFunction::ACT_TANH);
		addLayer(sigmoidlayer);
	}
	else if (layertype == LayerType::None) {
		ActivationLayer* sigmoidlayer = new ActivationLayer(num_of_node);
		sigmoidlayer->setActivationFunction(ActivationFunction::ACT_NONE);
		addLayer(sigmoidlayer);
	}

	else if (layertype == LayerType::Lost_MES) {
		LostLayer* lostlayer = new LostLayer(LostFunctionType::MSE, num_of_node);
		addLayer(lostlayer);
		
	}
}

void ArtificialNeuralNetwork::forwardPropagation()
{
	for (LayerNode* ptr = head; ; ptr = ptr->nextNode) {
		if (ptr == nullptr) break;
		ptr->layer->forwardPropagation();
	}
}

InputLayer * ArtificialNeuralNetwork::getInputLayer()
{
	if (head->layer->getLayerType() == LayerType::Input) {
		return dynamic_cast<InputLayer*>(head->layer);
	}
	return nullptr;
}

decimal ArtificialNeuralNetwork::getLastLost()
{
	return *(tail->layer->getZ());
}

decimal ArtificialNeuralNetwork::getLastPrediction()
{
	return *(tail->previousNode->layer->getZ());
}

void ArtificialNeuralNetwork::resetInputOutput()
{
	head->layer->resetLayer();
	tail->layer->resetLayer();
}

void ArtificialNeuralNetwork::NextStepOneByOne()
{
	head->layer->Next();
	tail->layer->Next();
}

decimal ArtificialNeuralNetwork::backPropagationOneByOne()
{
	tail->layer->backPropagation(nullptr);

	return 0;
}

void ArtificialNeuralNetwork::trainOneByOne(int numofiteration)
{
	resetInputOutput();
	for (int n = 0; n < numofiteration; ++n, NextStepOneByOne()) {
		forwardPropagation();
		backPropagationOneByOne();

		//printf("Loss = %f\n", getLastLost());
	}

}

void ArtificialNeuralNetwork::trainBatch(int batchSize)
{

}

void ArtificialNeuralNetwork::setUpfortest()
{
	// input
	LayerNode* ptr = head;
	dynamic_cast<InputLayer*>(ptr->layer)->setTestDataset();
	decimal* bi = dynamic_cast<InputLayer*>(ptr->layer)->getBiasWeight();
	bi[0] = 0.35f;
	bi[1] = 0.35f;

	// fc 1
	ptr = ptr->nextNode;
	FullyConnectedLayer* f1 = dynamic_cast<FullyConnectedLayer*>(ptr->layer);
	decimal* w1 = f1->getWeight();
	w1[0] = 0.15f;
	w1[1] = 0.25f;
	w1[2] = 0.20f;
	w1[3] = 0.30f;
	decimal* b1 = f1->getBiasWeight();
	b1[0] = 0.60f;
	b1[1] = 0.60f;

	// sigmoid 1
	ptr = ptr->nextNode;

	// fc 2
	ptr = ptr->nextNode;
	FullyConnectedLayer* f2 = dynamic_cast<FullyConnectedLayer*>(ptr->layer);
	decimal* w2 = f2->getWeight();
	w2[0] = 0.40f;
	w2[1] = 0.50f;
	w2[2] = 0.45f;
	w2[3] = 0.55f;
	decimal* b2 = f2->getBiasWeight();
	b2[0] = 0.00f;
	b2[1] = 0.00f;

	linkLostLayer();

}

void ArtificialNeuralNetwork::linkLostLayer()
{
	dynamic_cast<LostLayer*>(tail->layer)->setInput(dynamic_cast<InputLayer*>(head->layer));
}

void ArtificialNeuralNetwork::printAllWeight()
{
	printf("--------------------\n");
	for (LayerNode* ptr = head; ; ptr = ptr->nextNode) {
		if (ptr == nullptr) break;
		if (ptr->layer->getLayerType() == LayerType::FullyConnected) {
			decimal* weight = dynamic_cast<FullyConnectedLayer*>(ptr->layer)->getWeight();
			for (int i = 0; i < ptr->layer->getLayerSize(); ++i) {
				for (int j = 0; j < ptr->layer->getLastLayer()->getLayerSize(); ++ j)
					printf("%f ", *weight++);
				printf("\n");
			}
			printf("\n-------\n");
		}
	}
}

decimal ArtificialNeuralNetwork::simpleEvaluateAll(int datasetsize)
{
	decimal sum_lost = 0;

	resetInputOutput();
	for (int n = 0; n < datasetsize; ++n, NextStepOneByOne()) {
		forwardPropagation();
		sum_lost += getLastLost();
	}
	return sum_lost / decimal(datasetsize);
}

void ArtificialNeuralNetwork::writeDownData(char * path, int numofiteration)
{
	int output_size = tail->layer->getLayerSize();

	FILE* result_file;
	fopen_s(&result_file, path, "w");
	char num[20];

	for (int i = 0; i < sizeof(num); ++i) {
		num[i] = '\0';
	}

	resetInputOutput();
	for (int n = 0; n < numofiteration; ++n, NextStepOneByOne()) {
		forwardPropagation();
		backPropagationOneByOne();

		decimal predict = getLastPrediction();

		sprintf_s<sizeof(num)>(num, "%f\n", predict);

		fwrite(&num, sizeof(num), 1, result_file);
		//printf("Loss = %f\n", getLastLost());
	}
	fclose(result_file);
}
