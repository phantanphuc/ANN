#ifndef __LOSTLAYER_MSE_H__
#define __LOSTLAYER_MSE_H__

#include "BaseLayer.h"
#include "InputLayer.h"

enum LostFunctionType {
	MSE
};

class LostLayer : public BaseLayer {
public:
	LostLayer(LostFunctionType type, int layersize);
	~LostLayer();

	LostFunctionType lostfunctiontype;

	void forwardPropagation();
	void backPropagation(decimal* delta);

	void setInput(InputLayer* input);
	void setInput(decimal* dataset, int stride, int targetsize);

	void Next();
	void resetLayer();
private:
	decimal* target_ptr;
	decimal* dataset_ptr;

	decimal LostValue = 0;

	int stride;
	int target_size;
};
#endif
