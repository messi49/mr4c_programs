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

//cuda
const int N = 16; 
const int blocksize = 16; 
 
__global__ 
void hello(char *a, int *b) 
{
		a[threadIdx.x] += b[threadIdx.x];
}

using namespace MR4C;

//extend the Algorithm class                          
class ImageInfo : public Algorithm {
	public:

		//virtual method that will be executed                                                                                                           
		void executeAlgorithm(AlgorithmData& data, AlgorithmContext& context) {

			//****define algorithm here****

			//open native message block	
			std::string nativeHdr="\n*************************NATIVE_OUTPUT*************************\n"; 
			std::cout<<nativeHdr<<std::endl;

			while(1){
				//cuda
				char a[N] = "Hello \0\0\0\0\0\0";
				int b[N] = {15, 10, 6, 0, -11, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

				char *ad;
				int *bd;
				const int csize = N*sizeof(char);
				const int isize = N*sizeof(int);

				printf("%s", a);

				cudaMalloc( (void**)&ad, csize ); 
				cudaMalloc( (void**)&bd, isize ); 
				cudaMemcpy( ad, a, csize, cudaMemcpyHostToDevice ); 
				cudaMemcpy( bd, b, isize, cudaMemcpyHostToDevice ); 

				sleep(1);

				dim3 dimBlock( blocksize, 1 );
				dim3 dimGrid( 1, 1 );
				hello<<<dimGrid, dimBlock>>>(ad, bd);
				cudaMemcpy( a, ad, csize, cudaMemcpyDeviceToHost ); 
				cudaFree( ad );
				cudaFree( bd );

				printf("%s\n", a);
			}

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
