//std
#include <iostream>
#include <string>
#include <cstring>
#include <stdlib.h> //exit()
//mr4c
#include "algo_dev_api.h"
#include "mr4c_geo_api.h"

using namespace MR4C;

//extend the Algorithm class                          
class Map : public Algorithm 
{
public:

	//virtual method that will be executed                                                                                                           
	void executeAlgorithm(AlgorithmData& data, AlgorithmContext& context) 
	{
		//add your code here

		//open native message block	
		std::string nativeHdr="\n*************************NATIVE_OUTPUT*************************\n"; 
		std::cout<<nativeHdr<<std::endl;

		//print hello world message to stdout    
		std::cout << "  Hello World!!" << std::endl;
		printf("Hello messi!!\n");

		context.log( MR4C::Logger::INFO, "Hello messi!!!");

		//close message block		
		std::cout<<nativeHdr<<std::endl; 
	}

	//method that's called when algorithm is registered                                                                                                                                                        
	//list input and output datasets here                                                                                                                                                                      
	static Algorithm* create() 
	{
		Map* algo = new Map();
		return algo;
	}

};

//this will create a global variable that registers the algorithm when its library is loaded.                                                                                                               
MR4C_REGISTER_ALGORITHM(map,Map::create());

