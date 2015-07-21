//std
#include <iostream>
#include <string>
#include <cstring>
#include <stdlib.h> //exit()
//mr4c
#include "algo_dev_api.h"

using namespace MR4C;

typedef struct WordCount{
	char * word;
	int count;
};

//extend the Algorithm class
class Reduce : public Algorithm {
	public:

		//virtual method that will be executed                                                                                                           
		void executeAlgorithm(AlgorithmData& data, AlgorithmContext& context) 
		{		

			//****define algorithm here****
			//open native message block	
			std::string nativeHdr="\n*************************NATIVE_OUTPUT*************************\n"; 
			std::cout<<nativeHdr<<std::endl;

			//open files
			// input/output directory names specified in configuration file
			Dataset* input = data.getInputDataset("hist");
			Dataset* output = data.getOutputDataset("summaryOut");

			//get keyspace elements (files in directory by dimension)
			Keyspace keyspace = data.getKeyspace();
			std::vector<DataKeyElement> names = keyspace.getKeyspaceDimension(DataKeyDimension("NAME")).getElements();

			std::vector<WordCount> counter;
			int p = 0;

			//... iterate through keys and do work ...
			for ( std::vector<DataKeyElement>::iterator n=names.begin(); n != names.end(); n++ )
			{
				//print dimension name to stdout
				std::cout<<*n<<std::endl;

				//get input data
				DataKey key = *n;
				DataFile* myFile = input->getDataFile(key);
				char * fileBytes = myFile->getBytes();


				char *tok;
				WordCount wc;
				char mark[] = " .,\n";
				int matchflag = 0;
				int k = 0;

				tok = strtok(fileBytes, mark);
				while( tok != NULL ){
					if(k%2 == 0){
						wc.word = tok;
					}
					else if(k%2 == 1){
						wc.count = atoi(tok);

						for(p=0; p < counter.size(); p++){
							if(strcmp(counter[p].word, wc.word) == 0){
								counter[p].count += wc.count;
								matchflag = 1;
								break;
							}
						}

						if(matchflag == 0){
							counter.push_back(wc);
						}
						matchflag = 0;
					}
					tok = strtok( NULL, mark);  /* 2回目以降 */
					k++;
				}

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

			//print new output file contents
			printf("%s", textout);

			//write summary to file
			DataKey histKey;
			DataFile* histData = new DataFile(textout, strlen(textout), "binary");
			output->addDataFile(histKey, histData);

			//close message block		
			std::cout<<nativeHdr<<std::endl;
		}


		//method that's called when algorithm is registered                                                                                                                                                        
		//list input and output datasets here                                                                                                                                                                      
		static Algorithm* create() 
		{
			Reduce* algo = new Reduce();
			algo->addInputDataset("hist");
			algo->addOutputDataset("summaryOut");
			return algo;
		}

};

//this will create a global variable that registers the algorithm when its library is loaded.                                                                                                               
MR4C_REGISTER_ALGORITHM(reduce,Reduce::create());

