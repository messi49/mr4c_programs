#include "algo_dev_api.h"
#include <iostream>
#include <string.h>
#include <stdlib.h>

using namespace MR4C;

const int N = 20;

__global__
void count_word(char *a,int *b, int *c, int size) 
{
	int i = 0, k = 0;

	if(threadIdx.x < size){
		for(i = 0; i < size; i++){
			if(threadIdx.x != i){
				for(k = 0; k < N; k++){
					if(a[N * threadIdx.x + k] != a[N * i + k]){
						break;
					}
					if(k == N - 1){
						c[threadIdx.x] += b[i];
					}
				}
			}
		}
	}
	else{
		c[threadIdx.x] = -1;
	}
}

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
		for ( std::set<DataKey>::iterator itr = keys.begin(); itr != keys.end(); itr++ ) {	
			DataKey myKey = *itr;
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
			std::cout<<"  original file contents: \n"<<fileBytes;

			char *tok;
			char *word_ary;
			int *count_ary;
			char mark[] = " .,\n";
			int size = 0;
			int i = 0, k = 0;
			int allocate_size = N;

			// allocate memory
			word_ary = (char *)malloc(sizeof(char) * N * N);
			count_ary = (int *)malloc(sizeof(int) * N);
			memset(word_ary, 0, sizeof(char) * N * N);
			memset(count_ary, 0, sizeof(int) * N);

			// input words
			tok = strtok(fileBytes, mark);
			while( tok != NULL ){
				// reallocate memory
				if(size == allocate_size - 1){
					allocate_size += N;
					word_ary = (char *)realloc(word_ary, sizeof(char) * allocate_size * N);
					count_ary = (int *)realloc(count_ary, sizeof(int) * allocate_size);
					memset(&word_ary[N * size], 0, sizeof(char) * (allocate_size - size) * N);
				}

				// input word
				if(i%2 == 0){
					strcpy(&word_ary[size * N], tok);
				}

				//input count
				else if(i%2 == 1){
					count_ary[size] = atoi(tok);
					size++;
				}

				tok = strtok( NULL, mark);
				i++;
			}

			char *ad;
			int *bd;
			int *cd;

			int *count;
			count = (int *)malloc(sizeof(int) * size);

			const int csize = N * size * sizeof(char);
			const int isize = size * sizeof(int);

			cudaMalloc( (void**)&ad, csize ); 
			cudaMalloc( (void**)&bd, isize );
			cudaMalloc( (void**)&cd, isize );

			cudaMemcpy( ad, word_ary, csize, cudaMemcpyHostToDevice ); 
			cudaMemcpy( bd, count_ary, isize, cudaMemcpyHostToDevice ); 
			cudaMemcpy( cd, count_ary, isize, cudaMemcpyHostToDevice ); 

			dim3 dimBlock( size, 1 );
			dim3 dimGrid( 1, 1 );
			count_word<<<dimGrid, dimBlock>>>(ad, bd, cd, size);
			cudaMemcpy( count, cd, isize, cudaMemcpyDeviceToHost ); 
			cudaFree( ad );
			cudaFree( bd );
			cudaFree( cd );

			char answer_words[size][N];
			int answer_count[size];

			int num = 0;
			int dismatchflag = 0;

			for(i = 0; i < size; i++){
				if(count[i] == -1){
					break;
				}

				if(count[i] == 1 || count[i] == count_ary[i]){
					strncpy(answer_words[num], &word_ary[N * i], N);
					answer_count[num] = count[i];
					num++;
				}
				else{
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

			for(i=0; i < num; i++){
				sprintf(buf, "%s,%d\n",answer_words[i], answer_count[i]);
				strcat(textout, buf);
				memset(buf, 0, sizeof(buf));
			}

			//print new output file contents
			printf("\nAns: \n");
			printf("%s", textout);
			//std::cout<<"  output file contents: "<<fileBytes;

			//close message block		
			std::cout<<nativeHdr<<std::endl;
			
			// output file in the output folder
			Dataset* output = data.getOutputDataset("imageOut");
			DataFile* fileData = new DataFile(textout, strlen(textout), "testOut.bin");
			output->addDataFile(myKey, fileData);

			//free(word_ary);
			//free(textout);
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
