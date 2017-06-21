#include "BaseLayer.h"

BaseLayer::BaseLayer(int layersize){
	layer_size = layersize;
}

BaseLayer::~BaseLayer(){
	delete bias_weight;
	if (gradien)
		delete gradien;
}

int BaseLayer::getLayerSize()
{
	return layer_size;
}

LayerType BaseLayer::getLayerType()
{
	return layer_type;
}

decimal * BaseLayer::getBiasWeight()
{
	return bias_weight;
}

void BaseLayer::setPreviousLayer(BaseLayer * previous)
{
	previous_layer_ref = previous;

	previous->initBias(layer_size);

	gradien = new decimal[previous->getLayerSize() * layer_size];

	delta = new decimal[previous->getLayerSize()];

}

void BaseLayer::initBias(int size)
{
	bias_weight = new decimal[size];

	srand(time(NULL));
	for (int i = 0; i < size; ++i) {
		bias_weight[i] = rand() / float(RAND_MAX) * RANDOM_FACTOR;
	}
}

