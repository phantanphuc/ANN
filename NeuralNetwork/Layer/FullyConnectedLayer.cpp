#include "FullyConnectedLayer.h"

FullyConnectedLayer::FullyConnectedLayer(int layersize, int previousLayerSize) : BaseLayer(layersize){
	layer_type = LayerType::FullyConnected;
	weightsize = layersize * previousLayerSize;
	weight = new decimal[weightsize];
	//decimal weight[10];
	Z_vector = new decimal[layersize];

	srand(time(NULL));
	for (int i = 0; i < weightsize; ++i) {
		weight[i] = rand() / float(RAND_MAX) * RANDOM_FACTOR;
	}

}

FullyConnectedLayer::~FullyConnectedLayer(){
	delete weight;
	delete Z_vector;
}

void FullyConnectedLayer::forwardPropagation()
{
	int weightWidth = getWeightWidth();
	decimal* weightptr = weight;
	decimal* input = previous_layer_ref->getZ();

	/*
	for (int i = 0; i < layer_size; ++i) {
	decimal sum = 0;
	for (int j = 0; j < weightWidth; ++j) {
	sum += *(weightptr + j) * *(input + j);
	}
	Z_vector[i] = sum;
	weightptr += weightWidth;
	}
	*/

	int last_layer_size = previous_layer_ref->getLayerSize();

	for (int i = 0; i < weightWidth; ++i) {
		Z_vector[i] = 0;
	}

	for (int i = 0; i < last_layer_size; ++i) {
		
		for (int j = 0; j < layer_size; ++j) {
			Z_vector[j] += input[i] * *weightptr++;
		}
	}

	decimal* bias = previous_layer_ref->getBiasWeight();
	for (int i = 0; i < layer_size; ++i) {
		Z_vector[i] += bias[i];
	}

}

int FullyConnectedLayer::getWeightWidth()
{
	return previous_layer_ref->getLayerSize();
}

void FullyConnectedLayer::backPropagation(decimal * delta)
{
	int last_layer_size = previous_layer_ref->getLayerSize();


	///////////////////////////////////////
	//////// dz/da /////////////////////
	///////////////////////////////////////
	//  dz     sum(wi)
	// ---- = --------- = i
	//  dw       dw
	///////////////////////////////////////



	///////////////////////////////////////
	//////// dz/dw /////////////////////
	///////////////////////////////////////
	//  dz     sum(wi)
	// ---- = --------- = i
	//  dw       dw
	///////////////////////////////////////

	decimal* input = previous_layer_ref->getZ();
	decimal* gradien_ptr = gradien;

	//    w11 w12
	//    w21 w22
	//    w31 w32
	//    w41 w42

	for (int i = 0; i < last_layer_size; ++i) { // w11 w21 w31 ... wn1 V
		for (int j = 0; j < layer_size; ++j) { // w11 w12 w13 ... w1n ->
			*gradien_ptr++ = input[i] * delta[j];
		}
	}
	/////// UPDATE ////////
	/**/updateWeight();/**/
	///////////////////////
	previous_layer_ref->backPropagation(this->delta);
}

void FullyConnectedLayer::updateWeight()
{
	for (int i = 0; i < weightsize; ++i) {
		// W =  W -        eta                            *  dL/dw 
		weight[i] -= param_manager_ref->getLearningRate() * gradien[i];
	}
}

