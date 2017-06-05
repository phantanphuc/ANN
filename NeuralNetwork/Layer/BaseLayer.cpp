#include "BaseLayer.h"

BaseLayer::BaseLayer(int layersize){
	layer_size = layersize;
}

BaseLayer::~BaseLayer(){

}

int BaseLayer::getLayerSize()
{
	return layer_size;
}
