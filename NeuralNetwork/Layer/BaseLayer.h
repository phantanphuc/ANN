#ifndef __BASELAYER_H__
#define __BASELAYER_H__

#include <random>
#include <time.h>
#include "System\ParamManager.h"

#define RANDOM_FACTOR 0.001f

typedef float decimal;

enum LayerType {
	Input, FullyConnected, Sigmoid, Output, Lost_MES
};

class BaseLayer{
public:
	BaseLayer(int layersize);
	virtual ~BaseLayer();

	int getLayerSize();
	LayerType getLayerType();

	virtual decimal* getZ() { return Z_vector; };
	virtual void forwardPropagation() = 0;
	virtual void backPropagation(decimal* delta){}
	virtual void calculateDelta(){}

	decimal* getBiasWeight();

	virtual void setPreviousLayer(BaseLayer* previous);
	virtual void initBias(int size);

	virtual void resetLayer() {};

	void setParamManagerRef(ParamManager* paramManager) { param_manager_ref = paramManager; };

protected:
	/*
	Num of neural in layer (input vector size)
	*/
	int layer_size;

	LayerType layer_type;
	BaseLayer* previous_layer_ref = nullptr;
	

	decimal bias_value = 1;
	decimal* bias_weight;

	decimal* Z_vector;

	/*
	Back prop
	*/

	decimal* gradien = nullptr;
	decimal* delta = nullptr;

	ParamManager* param_manager_ref = nullptr;
};
#endif
