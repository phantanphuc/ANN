#ifndef __FULLYCONNECTEDLAYER_H__
#define __FULLYCONNECTEDLAYER_H__

#include "BaseLayer.h"

class FullyConnectedLayer : public BaseLayer {
public:
	FullyConnectedLayer(int layersize, int previousLayerSize);
	~FullyConnectedLayer();
private:
};
#endif
