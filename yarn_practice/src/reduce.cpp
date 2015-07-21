//std
#include <iostream>
#include <string>
#include <cstring>
#include <stdlib.h> //exit()
//mr4c
#include "algo_dev_api.h"

using namespace MR4C;

//extend the Algorithm class
class Reduce : public Algorithm {
public:

	//virtual method that will be executed                                                                                                           
	void executeAlgorithm(AlgorithmData& data, AlgorithmContext& context) 
	{		
		
		//****define algorithm here****
		//open native message block	
		std::string nativeHdr="\n*************************NATIVE_OUTPUT[Reduce]*************************\n"; 
		std::cout<<nativeHdr<<std::endl;

		//close message block		
		std::cout<<nativeHdr<<std::endl;
	}

	//method that's called when algorithm is registered                                                                                                                                                        
	//list input and output datasets here                                                                                                                                                                      
	static Algorithm* create() 
	{
		Reduce* algo = new Reduce();
		//algo->addInputDataset("hist");
		//algo->addOutputDataset("summaryOut");
		return algo;
	}

};

//this will create a global variable that registers the algorithm when its library is loaded.                                                                                                               
MR4C_REGISTER_ALGORITHM(reduce,Reduce::create());

