#include "ArtificialNeuralNetwork.h"

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
		SigmoidLayer* sigmoidlayer = new SigmoidLayer(num_of_node);
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

void ArtificialNeuralNetwork::resetInputOutput()
{
	head->layer->resetLayer();
	tail->layer->resetLayer();
}

decimal ArtificialNeuralNetwork::backPropagationOneByOne()
{
	tail->layer->backPropagation(nullptr);

	return 0;
}

void ArtificialNeuralNetwork::trainOneByOne()
{
	
	
	resetInputOutput();
	forwardPropagation();
	
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

	dynamic_cast<LostLayer*>(tail->layer)->setInput(dynamic_cast<InputLayer*>(head->layer));

}
