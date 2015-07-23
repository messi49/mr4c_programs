//std
#include <iostream>
#include <string>
#include <cstring>
#include <stdlib.h> //exit()
//mr4c
#include "algo_dev_api.h"
#include "mr4c_geo_api.h"

using namespace MR4C;

const int N = 20;

	__global__
void count_word(char *a, int *c, int size) 
{
	int i = 0, k = 0;
	int count = 1;

	if(threadIdx.x < size){

		for(i = 0; i < size; i++){
			if(threadIdx.x != i){
				for(k = 0; k < N; k++){
					if(a[N * threadIdx.x + k] != a[N * i + k]){
						break;
					}
					if(k == N - 1){
						count++;
					}
				}
			}
		}

		c[threadIdx.x] = count;
	}
	else{
		c[threadIdx.x] = -1;
	}
}

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
			for ( std::vector<DataKeyElement>::iterator itr = names.begin(); itr != names.end(); itr++ )
			{
				//print dimension name to stdout
				std::cout<<*itr<<std::endl;

				//get multispectral data
				DataKey skyKey = *itr;
				DataFile* skyFile = input->getDataFile(skyKey);

				int fileSize=skyFile->getSize();
				char * fileBytes=skyFile->getBytes();

				//open native message block	
				std::string nativeHdr="\n*************************NATIVE_OUTPUT*************************\n"; 
				std::cout<<nativeHdr<<std::endl;		

				//report image info to stdout
				std::cout<<"  Image Loaded"<<std::endl;
				std::cout<<"  "<<fileSize<<" bytes"<<std::endl;

				//print original file contents
				std::cout<<"  original file contents: "<<fileBytes;

				char *tok;
				char *word_ary;
				char mark[] = " .,\n";
				int size = 0;
				int i = 0, k = 0;
				int allocate_size = N;

				// allocate memory
				word_ary = (char *)malloc(sizeof(char) * N * N);
				memset(word_ary, 0, sizeof(char) * N * N);

				// input words
				tok = strtok(fileBytes, mark);
				while( tok != NULL ){
					// reallocate memory
					if(size == allocate_size - 5){
						allocate_size += N;
						word_ary = (char *)realloc(word_ary, sizeof(char) * allocate_size * N);
						memset(&word_ary[N * size], 0, sizeof(char) * (allocate_size - size) * N);
					}
					strcpy(&word_ary[size * N], tok);
					tok = strtok( NULL, mark);
					size++;
				}

				size--;

				char *ad;
				int *cd;

				int *count;
				count = (int *)malloc(sizeof(int) * size);

				const int csize = N * size * sizeof(char);
				const int isize = size * sizeof(int);

				cudaMalloc( (void**)&ad, csize ); 
				cudaMalloc( (void**)&cd, isize );

				cudaMemcpy( ad, word_ary, csize, cudaMemcpyHostToDevice ); 

				dim3 dimBlock( size, 1 );
				dim3 dimGrid( 1, 1 );
				count_word<<<dimGrid, dimBlock>>>(ad, cd, size);
				cudaMemcpy( count, cd, isize, cudaMemcpyDeviceToHost ); 
				cudaFree( ad );
				cudaFree( cd );

				char answer_words[size][N];
				int answer_count[size];

				int num = 0;
				int dismatchflag = 0;

				for(i = 0; i < size; i++){
					if(count[i] == -1){
						break;
					}

					if(count[i] == 1){
						strncpy(answer_words[num], &word_ary[N * i], N);
						answer_count[num] = count[i];
						num++;
					}
					else if(count[i] > 1){
						for(k = 0; k < num; k++){
							if(strncmp(&word_ary[N * i], answer_words[k], N) == 0){
								dismatchflag = 1;
								break;
							}
						}
						if(dismatchflag == 0){
							strncpy(answer_words[num], &word_ary[N * i], N);
							answer_count[num] = count[i];
							num++;
						}
						else{
							dismatchflag = 0;
						}
					}
				}


				char * textout = (char *)malloc(sizeof(char) * (num * N + 2));
				memset(textout, 0, sizeof(char) * (num * N + 2));
				char buf[N + 2];

				for(i=0; i < num - 1; i++){
					sprintf(buf, "%s,%d\n",answer_words[i], answer_count[i]);
					strcat(textout, buf);
					memset(buf, 0, sizeof(buf));
				}

				//print new output file contents
				//printf("%s\n", textout);
				//std::cout<<"  output file contents: "<<fileBytes;

				//close message block		
				std::cout<<nativeHdr<<std::endl;

				//output histogram
				DataKey histKey = DataKey(*itr);
				DataFile* histData = new DataFile(textout, strlen(textout), "text/plain");
				outputHist->addDataFile(histKey, histData);

				//free(word_ary);
				//free(textout);

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

