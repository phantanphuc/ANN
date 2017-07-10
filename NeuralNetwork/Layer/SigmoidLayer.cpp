#include "SigmoidLayer.h"

ActivationLayer::ActivationLayer(int size) : BaseLayer(size) {
	Z_vector = new decimal[size];
}

ActivationLayer::~ActivationLayer(){
	delete Z_vector;
}


void ActivationLayer::forwardPropagation()
{
	decimal* previous_layer_z = previous_layer_ref->getZ();

	if (activation_function == ActivationFunction::ACT_SIGMOID) {
		for (int i = 0; i < layer_size; ++i) {
			Z_vector[i] = SIGMOID(previous_layer_z[i]);
		}
	}
	else if (activation_function == ActivationFunction::ACT_TANH) {
		for (int i = 0; i < layer_size; ++i) {
			Z_vector[i] = TANH(previous_layer_z[i]);
		}
	}
	else if (activation_function == ActivationFunction::ACT_NONE) {
		for (int i = 0; i < layer_size; ++i) {
			Z_vector[i] = previous_layer_z[i];
		}
	}
}

void ActivationLayer::setPreviousLayer(BaseLayer * previous)
{
	previous_layer_ref = previous;

	delta = new decimal[layer_size];
}

void ActivationLayer::initBias(int size)
{
	
	previous_layer_ref->initBias(size);

	bias_weight = previous_layer_ref->getBiasWeight();
}

void ActivationLayer::backPropagation(decimal * delta)
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

	if (activation_function == ActivationFunction::ACT_SIGMOID) {
		for (int i = 0; i < layer_size; ++i) {
			decimal da_dz = SIGMOID(z[i]) * (1 - SIGMOID(z[i]));
			this->delta[i] *= da_dz;
		}
	}
	else if (activation_function == ActivationFunction::ACT_TANH) {
		for (int i = 0; i < layer_size; ++i) {
			decimal TanhValue = TANH(z[i]);
			decimal da_dz = 1 - TanhValue * TanhValue;
			this->delta[i] *= da_dz;
		}
	} 
	else if (activation_function == ActivationFunction::ACT_NONE) {
		
	}



	previous_layer_ref->backPropagation(this->delta);
}

void ActivationLayer::setActivationFunction(ActivationFunction type)
{
	activation_function = type;
}
