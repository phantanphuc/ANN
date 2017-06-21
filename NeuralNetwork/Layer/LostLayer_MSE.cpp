#include "LostLayer_MSE.h"

LostLayer::LostLayer(LostFunctionType type, int layersize) : BaseLayer(layersize) {
	Z_vector = new decimal[layersize];
	lostfunctiontype = type;

}

LostLayer::~LostLayer(){

}

void LostLayer::forwardPropagation()
{
	decimal* target = target_ptr;
	decimal* input = previous_layer_ref->getZ();
	LostValue = 0;

	decimal* z_ptr = Z_vector;

	if (lostfunctiontype == LostFunctionType::MSE) {
		for (int i = 0; i < target_size; ++i) {
			Z_vector[i] = (target[i] - input[i]) * (target[i] - input[i]) / 2.0f;
			LostValue += Z_vector[i];
		}
		//LostValue = LostValue / float(target_size);
	}
}

void LostLayer::backPropagation(decimal* delta)
{
	int last_layer_size = previous_layer_ref->getLayerSize();

	/*
	dLost   
	----- 
	 dW    
	*/

	/*
	delta:
	w11 w12
	w21 w22
	w31 w32

	Ow1
			Ow1
	Ow2
			Ow2
	Ow3
	
	*/

	///////////////////////////////////////
	//////// dLost/dW /////////////////////
	///////////////////////////////////////
	//         1                  |
	// Lost = --- ( (a - t )^2    | 
	//         2                  |
	//  dL     dL   da   dz
	// ---- = ---- ---- ----
	//  dW     da   dz   dw
	///////////////////////////////////////
	decimal* a = previous_layer_ref->getZ();
	//decimal* gradien_ptr = gradien;
	decimal* z_ptr = Z_vector;
	decimal* delta_ptr = this->delta;

	//  dL               |  da
	// ---- = (a - t)    | ---- =
	//  da               |  dz

	for (int i = 0; i < last_layer_size; ++i) {
		*delta_ptr++ = (a[i] - target_ptr[i]);
	}
	previous_layer_ref->backPropagation(this->delta);
}


void LostLayer::setInput(InputLayer * input)
{
	dataset_ptr = input->getDataSet();
	stride = input->getDataStride();
	target_size = input->getTargetSize();
	target_ptr = dataset_ptr + (stride - target_size);
}

void LostLayer::setInput(decimal * dataset, int stride, int targetsize)
{
	dataset_ptr = dataset;
	target_ptr = dataset + (stride - target_size);
	this->stride = stride;
	this->target_size = targetsize;
}

void LostLayer::Next()
{
	target_ptr += stride;
}

void LostLayer::resetLayer()
{
	target_ptr = dataset_ptr + (stride - target_size);
}
