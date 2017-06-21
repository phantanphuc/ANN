#ifndef __SIGMOIDLAYER_H__
#define __SIGMOIDLAYER_H__

#include "BaseLayer.h"

#define SIGMOID(x) 1/(1.0 + exp(-x))

class SigmoidLayer : public BaseLayer {
public:
	SigmoidLayer(int size);
	~SigmoidLayer();

	void forwardPropagation();

	void setPreviousLayer(BaseLayer* previous);
	void initBias(int size);

	void backPropagation(decimal* delta);

private:

};
#endif
