#include "SigmoidLayer.h"

SigmoidLayer::SigmoidLayer(int size) : BaseLayer(size) {
	Z_vector = new decimal[size];
}

SigmoidLayer::~SigmoidLayer(){
	delete Z_vector;
}


void SigmoidLayer::forwardPropagation()
{
	decimal* previous_layer_z = previous_layer_ref->getZ();

	for (int i = 0; i < layer_size; ++i) {
		Z_vector[i] = SIGMOID(previous_layer_z[i]);
	}
}

void SigmoidLayer::setPreviousLayer(BaseLayer * previous)
{
	previous_layer_ref = previous;

	delta = new decimal[layer_size];
}

void SigmoidLayer::initBias(int size)
{
	
	previous_layer_ref->initBias(size);

	bias_weight = previous_layer_ref->getBiasWeight();
}

void SigmoidLayer::backPropagation(decimal * delta)
{
	int last_layer_size = previous_layer_ref->getLayerSize();
	memcpy_s(this->delta, sizeof(decimal) * layer_size, delta, sizeof(decimal) * layer_size);

	///////////////////////////////////////
	//////// da/dz /////////////////////
	///////////////////////////////////////
	//  da     sigmoid(z)
	// ---- = ------------ = sigmoid(z) (1 - sigmoid(z))
	//  dz         dz
	///////////////////////////////////////

	decimal* z = previous_layer_ref->getZ();

	for (int i = 0; i < layer_size; ++i) {
		decimal da_dz = SIGMOID(z[i]) * (1 - SIGMOID(z[i]));
		this->delta[i] *= da_dz;
	}

	previous_layer_ref->backPropagation(this->delta);
}
