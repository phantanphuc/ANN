#ifndef __PARAMMANAGER_H__
#define __PARAMMANAGER_H__

typedef float decimal;

class ParamManager {
public:
	ParamManager();
	~ParamManager();

	void setLearningRate(decimal lr) { learningRate = lr; };
	decimal getLearningRate() { return learningRate; };

private:
	
	decimal learningRate = 0.5;

};
#endif
