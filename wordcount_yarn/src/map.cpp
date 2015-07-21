//std
#include <iostream>
#include <string>
#include <cstring>
#include <stdlib.h> //exit()
//mr4c
#include "algo_dev_api.h"
#include "mr4c_geo_api.h"

using namespace MR4C;

typedef struct WordCount{
	char * word;
	int count;
};

//extend the Algorithm class                          
class Map : public Algorithm 
{
public:

	//virtual method that will be executed                                                                                                           
	void executeAlgorithm(AlgorithmData& data, AlgorithmContext& context) 
	{		
		//****define algorithm here****
		//open native message block	
		std::string nativeHdr="\n*************************NATIVE_OUTPUT*************************\n"; 
		std::cout<<nativeHdr<<std::endl;

		//open image files
		// input/output directory names specified in configuration file
		Dataset* input = data.getInputDataset("imagesIn");
		Dataset* outputHist = data.getOutputDataset("hist");
		
		//get keyspace elements (files in directory by dimension)
		Keyspace keyspace = data.getKeyspace();
		std::vector<DataKeyElement> names = keyspace.getKeyspaceDimension(DataKeyDimension("NAME")).getElements();
				
		
		//... iterate through keys and do work ...
		for ( std::vector<DataKeyElement>::iterator n=names.begin(); n != names.end(); n++ )
		{
			//print dimension name to stdout
			std::cout<<*n<<std::endl;
			
			//get multispectral data
			DataKey skyKey = *n;
			DataFile* skyFile = input->getDataFile(skyKey);
			
			int fileSize=skyFile->getSize();
			char * fileBytes=skyFile->getBytes();

			std::vector<WordCount> counter;

			char *tok;
			char mark[] = " .,\n";
			int p = 0;
			int matchflag = 0;

			tok = strtok(fileBytes, mark);
			while( tok != NULL ){
				for(p=0; p < counter.size(); p++){
					if(strcmp(counter[p].word, tok) == 0){
						counter[p].count++;
						matchflag = 1;
						break;
					}
				}
				if(matchflag == 1){
					matchflag = 0;
				}
				else{
					if(strlen(tok) > 0)
						counter.push_back((WordCount){tok, 1});
				}
				tok = strtok( NULL, mark);  /* 2回目以降 */
			}

			size_t len = counter.size() * (20 + 2);
			char * textout = (char *)malloc(len + 1);
			memset(textout, 0, sizeof(textout));
			char buf[22];

			printf("Answer(size = %d): \n", counter.size());
			for(p=0; p < counter.size() - 1; p++){
				sprintf(buf, "%s,%d\n", counter[p].word, counter[p].count);
				strcat(textout, buf);
				memset(buf, 0, sizeof(buf));
			}

			//output histogram
			DataKey histKey = DataKey(*n);
			DataFile* histData = new DataFile(textout, strlen(textout), "text/plain");
			outputHist->addDataFile(histKey, histData);

			//close message block		
			std::cout<<nativeHdr<<std::endl;
		}
	}

	//method that's called when algorithm is registered                                                                                                                                                        
	//list input and output datasets here                                                                                                                                                                      
	static Algorithm* create() 
	{
		Map* algo = new Map();
		algo->addInputDataset("imagesIn");
		algo->addOutputDataset("hist");
		return algo;
	}

};

//this will create a global variable that registers the algorithm when its library is loaded.                                                                                                               
MR4C_REGISTER_ALGORITHM(map,Map::create());

