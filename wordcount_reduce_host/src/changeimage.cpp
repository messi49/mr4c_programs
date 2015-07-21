#include "algo_dev_api.h"
#include <iostream>
#include <string.h>
#include <stdlib.h>

using namespace MR4C;
using namespace std;

typedef struct WordCount{
	char * word;
	int count;
};

//extend the Algorithm class                          
class ChangeImage : public Algorithm {
public:

	//virtual method that will be executed                                                                                                           
	void executeAlgorithm(AlgorithmData& data, AlgorithmContext& context) {

		//****define algorithm here****

		//open image file
		// input name specified in configuration file
		Dataset* input = data.getInputDataset("imageIn");
		std::set<DataKey> keys = input->getAllFileKeys();
	
		//... iterate through input keys and do work ...
		for ( std::set<DataKey>::iterator i = keys.begin(); i != keys.end(); i++ ) {	
			DataKey myKey = *i;
			DataFile* myFile = input->getDataFile(myKey);
			int fileSize=myFile->getSize();
			char * fileBytes=myFile->getBytes();

			//open native message block	
			std::string nativeHdr="\n*************************NATIVE_OUTPUT*************************\n"; 
			std::cout<<nativeHdr<<std::endl;		
			
			//report image info to stdout
			std::cout<<"  Image Loaded"<<std::endl;
			std::cout<<"  "<<fileSize<<" bytes"<<std::endl;
			
			//print original file contents
			std::cout<<"  original file contents: "<<fileBytes;

			std::vector<WordCount> counter;

			char *tok;
			WordCount wc;
			char mark[] = " .,\n";
			int p = 0;
			int matchflag = 0;
			int k=0;

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
			//std::cout<<"  output file contents: "<<fileBytes;

			//close message block		
			std::cout<<nativeHdr<<std::endl;
			
			// output file in the output folder
			Dataset* output = data.getOutputDataset("imageOut");
			DataFile* fileData = new DataFile(textout, strlen(textout), "testOut.bin");
			output->addDataFile(myKey, fileData);
		}
	}

	//method that's called when algorithm is registered                                                                                                                                                        
	//list input and output datasets here                                                                                                                                                                      
	static Algorithm* create() {
		ChangeImage* algo = new ChangeImage();
		algo->addInputDataset("imageIn");
		algo->addOutputDataset("imageOut");
		return algo;
		}
	};

//this will create a global variable that registers the algorithm when its library is loaded.                                                                                                               
MR4C_REGISTER_ALGORITHM(changeimage,ChangeImage::create());
