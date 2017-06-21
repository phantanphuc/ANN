#ifndef __ARTIFICIALNEURALNETWORK_H__
#define __ARTIFICIALNEURALNETWORK_H__

#include "Layer\FullyConnectedLayer.h"
#include "Layer\InputLayer.h"
#include "Layer\SigmoidLayer.h"
#include "Layer\LostLayer_MSE.h"
#include "System\ParamManager.h"

struct LayerNode
{
	BaseLayer* layer;
	LayerNode* nextNode = nullptr;
	LayerNode* previousNode = nullptr;
};

class ArtificialNeuralNetwork {
public:
	ArtificialNeuralNetwork();
	~ArtificialNeuralNetwork();

	void addLayer(BaseLayer* layer);
	void addBasicLayer(LayerType layertype, int num_of_node);
	void forwardPropagation();

	InputLayer* getInputLayer();

	decimal getLastLost();

	void resetInputOutput();
	decimal backPropagationOneByOne();
	
	void trainOneByOne();
	void setUpfortest();

	///////////////// GETSET ////////////////////////
	void setLearningRate(decimal lr) { paramManager->setLearningRate(lr); };
	decimal getLearningRate() { return paramManager->getLearningRate(); };
private:
	LayerNode* head = nullptr;
	LayerNode* tail = nullptr;

	ParamManager* paramManager;
};
#endif
