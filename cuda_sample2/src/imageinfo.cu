//std
#include <iostream>
#include <stdlib.h> //exit()
#include <stdio.h>
//mr4c
#include "algo_dev_api.h"
//gdal
#include "gdal/gdal_priv.h"
#include "gdal/cpl_conv.h" // for CPLMalloc()
#include "gdal/cpl_string.h"

//vsi
#include "gdal/cpl_vsi.h"
#include "gdal/cpl_port.h"
#include <unistd.h>
#include <sys/stat.h>

#define MATRIX_SIZE 9

//cuda
const int N = 16; 
const int blocksize = 1; 
 
__global__ 
void add(int *a, int *b) 
{
		a[threadIdx.x] +=	1;
		b[threadIdx.x] += 1;
}


using namespace MR4C;

//extend the Algorithm class                          
class ImageInfo : public Algorithm {
	public:

		//virtual method that will be executed                                                                                                           
		void executeAlgorithm(AlgorithmData& data, AlgorithmContext& context) {

			//****define algorithm here****

			int matrix[MATRIX_SIZE][2];
			int matrix_num = 0;
			int counter = 0;

			//Input Data
			Dataset* input = data.getInputDataset("imageIn");
			std::set<DataKey> keys = input->getAllFileKeys();

			for ( std::set<DataKey>::iterator i = keys.begin(); i != keys.end(); i++ ) {	
				DataKey myKey = *i;
				DataFile* myFile = input->getDataFile(myKey);
				int fileSize=myFile->getSize();
				char * filebytes=myFile->getBytes();

				std::string nativehdr="\n*************************native_output*************************\n"; 
				std::cout<<nativehdr<<std::endl;	

				printf("fileSize = %d\n", fileSize);

				for (int b=0;b<fileSize-1;b++){
					if(filebytes[b] == ',' || filebytes[b] == '\n'){
					}
					else{
						matrix[counter][matrix_num] = filebytes[b];
						printf("%c,", matrix[counter][matrix_num]);
						counter++;
						if(counter == MATRIX_SIZE){
							matrix_num++;
						}
					}
				}
			}

			
			//open native message block	
			std::string nativeHdr="\n*************************NATIVE_OUTPUT*************************\n"; 
			std::cout<<nativeHdr<<std::endl;

			//while(1){
				//cuda
				int a[MATRIX_SIZE];
				int b[MATRIX_SIZE];

				for (int i = 0; i < MATRIX_SIZE; i++) {
					a[i]=matrix[i][0];
					b[i]=matrix[i][0];
				}

				int *ad;
				int *bd;
				const int isize = MATRIX_SIZE*sizeof(int);

				cudaMalloc( (void**)&ad, isize ); 
				cudaMalloc( (void**)&bd, isize ); 
				cudaMemcpy( ad, a, isize, cudaMemcpyHostToDevice ); 
				cudaMemcpy( bd, b, isize, cudaMemcpyHostToDevice ); 

				sleep(1);

				dim3 dimBlock( blocksize, 1 );
				dim3 dimGrid( 1, 1 );
				add<<<dimGrid, dimBlock>>>(ad, bd);
				cudaMemcpy( a, ad, isize, cudaMemcpyDeviceToHost ); 
				cudaFree( ad );
				cudaFree( bd );


				for (int i = 0; i < MATRIX_SIZE; i++) {
					printf("%d,", a[i]);
				}
			//}

			//close message block		
			std::cout<<nativeHdr<<std::endl;
		}

		//method that's called when algorithm is registered                                                                                                                                                        
		//list input and output datasets here                                                                                                                                                                      
		static Algorithm* create() {
			ImageInfo* algo = new ImageInfo();
			algo->addInputDataset("imageIn");
			return algo;
		}
};

//this will create a global variable that registers the algorithm when its library is loaded.                                                                                                               
MR4C_REGISTER_ALGORITHM(imageinfo,ImageInfo::create());
