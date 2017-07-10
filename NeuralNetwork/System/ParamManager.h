#ifndef __PARAMMANAGER_H__
#define __PARAMMANAGER_H__

typedef float decimal;

class ParamManager {
public:
	ParamManager();
	~ParamManager();

	void setLearningRate(decimal lr) { learningRate = lr; };
	decimal getLearningRate() { return learningRate; };

	void setBiasLearningRate(decimal lr) { BiasLearningRate = lr; };
	decimal getBiasLearningRate() { return BiasLearningRate; }

private:
	
	decimal learningRate = 0.5f;
	decimal BiasLearningRate = 0.005f;

};
#endif
