//std
#include <iostream>
#include <string>
#include <cstring>
#include <stdlib.h> //exit()
//mr4c
#include "algo_dev_api.h"
#include "mr4c_geo_api.h"


#define MATRIX_SIZE 9

using namespace MR4C;

//extend the Algorithm class                          
class Map : public Algorithm 
{
public:

	//virtual method that will be executed                                                                                                           
	void executeAlgorithm(AlgorithmData& data, AlgorithmContext& context) 
	{	

		int matrix[MATRIX_SIZE][2];
		int mtrix_num = 0;
		int counter = 0;

		Dataset* input = data.getInputDataset("matrixIn");
		std::set<DataKey> keys = input->getAllFileKeys();


		for ( std::set<DataKey>::iterator i = keys.begin(); i != keys.end(); i++ ) {	
			DataKey myKey = *i;
			DataFile* myFile = input->getDataFile(myKey);
			int fileSize=myFile->getSize();
			char * fileBytes=myFile->getBytes();

			std::string nativeHdr="\n*************************NATIVE_OUTPUT*************************\n"; 
			std::cout<<nativeHdr<<std::endl;	


			for (int b=0;b<fileSize-1;b++){
				if(fileBytes[b] == ',' || fileBytes[b] == '\n'){
				}
				else if(counter == (MATRIX_SIZE - 1)){
					break;
				}
				else{
					matrix[counter][matrix_num] = fileBytes[b];
					counter++;
				}
			}

			matrix_num++;
		}

	}

	//method that's called when algorithm is registered                                                                                                                                                        
	//list input and output datasets here                                                                                                                                                                      
	static Algorithm* create() 
	{
		Map* algo = new Map();
		algo->addInputDataset("matrixIn");
		algo->addOutputDataset("matrixOut");
		return algo;
	}

};

//this will create a global variable that registers the algorithm when its library is loaded.                                                                                                               
MR4C_REGISTER_ALGORITHM(map,Map::create());

