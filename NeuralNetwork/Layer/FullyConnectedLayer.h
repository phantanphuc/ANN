#ifndef __FULLYCONNECTEDLAYER_H__
#define __FULLYCONNECTEDLAYER_H__

#include "BaseLayer.h"



class FullyConnectedLayer : public BaseLayer {
public:
	FullyConnectedLayer(int layersize, int previousLayerSize);
	~FullyConnectedLayer();

	void forwardPropagation();

	int getWeightWidth();

	void backPropagation(decimal* delta);

	decimal* getWeight() { return weight; };

private:

	int weightsize;
	void updateWeight();

	decimal* weight;
	
};
#endif
