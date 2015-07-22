#include <stdio.h>

const int N = 20; 
const int blocksize = 20; 

	__global__ 
void hello(char *a, int *c, int size) 
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

int main()
{
	char words[N][N] =
	{
		"ABCDE",
		"xyz",
		"Hi",
		"japan",
		"xyz",
		"Hi",
		"cup",
		"paper",
		"Hi",
		"Apple"
	};

	int size = 10;

	int count[N];

	char answer_words[N][N];
	int answer_count[N];

	char *ad;
	int *cd;

	const int csize = N*N*sizeof(char);
	const int isize = N*sizeof(int);

	cudaMalloc( (void**)&ad, csize ); 
	cudaMalloc( (void**)&cd, isize );

	cudaMemcpy( ad, words, csize, cudaMemcpyHostToDevice ); 

	dim3 dimBlock( blocksize, 1 );
	dim3 dimGrid( 1, 1 );
	hello<<<dimGrid, dimBlock>>>(ad, cd, size);
	cudaMemcpy( count, cd, isize, cudaMemcpyDeviceToHost ); 
	cudaFree( ad );
	cudaFree( cd );

	int i = 0, k = 0;

	int num = 0;
	int dismatchflag = 0;

	for(i = 0; i < N; i++){
		if(count[i] == -1){
			break;
		}

		if(count[i] == 1){
			strcpy(answer_words[num], words[i]);
			answer_count[num] = count[i];
			num++;
		}
		else if(count[i] > 1){
			for(k = 0; k < num; k++){
				if(strcmp(words[i], answer_words[k]) == 0){
					dismatchflag = 1;
					break;
				}
			}
			if(dismatchflag == 0){
				strcpy(answer_words[num], words[i]);
				answer_count[num] = count[i];
				num++;
			}
			else{
				dismatchflag = 0;
			}
		}
	}

	for(i = 0; i < num; i++){
		printf("%s, %d\n", answer_words[i], answer_count[i]);
	}
	
	return EXIT_SUCCESS;
}
