#ifndef __SIGMOIDLAYER_H__
#define __SIGMOIDLAYER_H__

#include "BaseLayer.h"

#define SIGMOID(x) 1/(1.0 + exp(-x))
#define TANH(x) (1 - exp(-2*x)) / (1 + exp(-2*x))

enum ActivationFunction {
	ACT_NONE, ACT_SIGMOID, ACT_TANH
};

class ActivationLayer : public BaseLayer {
public:
	ActivationLayer(int size);
	~ActivationLayer();

	void forwardPropagation();

	void setPreviousLayer(BaseLayer* previous);
	void initBias(int size);

	void backPropagation(decimal* delta);
	void setActivationFunction(ActivationFunction type);

private:
	ActivationFunction activation_function;
};
#endif
