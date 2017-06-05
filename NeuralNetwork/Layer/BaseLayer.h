#ifndef __BASELAYER_H__
#define __BASELAYER_H__

class BaseLayer{
public:
	BaseLayer(int layersize);
	~BaseLayer();

	int getLayerSize();

protected:
	/*
	Num of neural in layer (input vector size)
	*/
	int layer_size;
};
#endif
