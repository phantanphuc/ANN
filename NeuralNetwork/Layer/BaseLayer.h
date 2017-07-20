#ifndef __BASELAYER_H__
#define __BASELAYER_H__

#include <random>
#include <iostream>
#include <time.h>
#include "System\ParamManager.h"

#define RANDOM_FACTOR 0.01f

typedef double decimal;

enum LayerType {
	Input, FullyConnected, Sigmoid, Tanh, None, Lost_MES
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
	virtual void Next() {};

	virtual void updateWeight() {};

	void setParamManagerRef(ParamManager* paramManager) { param_manager_ref = paramManager; };

	void setUpdateBias(bool isupdate) { isUpdateBias = isupdate; };
	BaseLayer* getLastLayer() { return previous_layer_ref; };

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
	decimal* gradien_bias = nullptr;
	decimal* delta = nullptr;
	bool isUpdateBias = false;

	ParamManager* param_manager_ref = nullptr;
};
#endif
