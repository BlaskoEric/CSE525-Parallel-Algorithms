/******************************************************************************
* Eric Blasko
* 6/02/19
* Homework #3
* RecMatMulTiled.cu
* This program performs rectangle matrix multiplication, which uses shared mem
* of size TILE_WIDTH x TILE_WIDTH. Values of Matrix M and N are chosen by the
* user such that M is of size JxK and N is of size KxL. The kernal function 
* produces the results of the matrix multiplication between M and N, storing
* it in matrix P which is of size JxL.  
******************************************************************************/
#include <stdio.h>
#include <assert.h>

#define TILE_WIDTH 4

//kernal to compute C = A * B. Uses shared memory/tile execution
__global__ void MatrixMulKernel(float *d_M, float *d_N, float *d_P, int JSIZE, int KSIZE, int LSIZE)
{

    //tile size to store element in shared memory
    __shared__ float Mds[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Nds[TILE_WIDTH][TILE_WIDTH];

    //generate ids of threads and blocks
    int bx = blockIdx.x; int by = blockIdx.y;
    int tx = threadIdx.x; int ty = threadIdx.y;

    // Identify row, col of d_P elem to work on
    int Row = by*TILE_WIDTH + ty;
    int Col = bx*TILE_WIDTH + tx; 

    float Pvalue = 0.0;

    // loop over d_M and dN tiles required to compute d_P elem
    for (int ph = 0; ph < ((KSIZE - 1)/TILE_WIDTH) + 1; ph++) {
 
         //collaborative loading of d_M and d_N tiles into shared memory       
        if((Row < JSIZE) && (ph*TILE_WIDTH + tx) < KSIZE)
            Mds[ty][tx] = d_M[Row*KSIZE + ((ph*TILE_WIDTH) + tx)];
        else
            Mds[ty][tx] = 0.0;
        if((ph*TILE_WIDTH + ty) < KSIZE && Col < LSIZE)
            Nds[ty][tx] = d_N[(ph*TILE_WIDTH + ty) * LSIZE + Col];
        else
            Nds[ty][tx] = 0.0;
 
        __syncthreads();

        for (int k = 0; k < TILE_WIDTH;k++)
            Pvalue += Mds[ty][k] * Nds[k][tx];
            
        __syncthreads();
    }

    //Store final results into C
    if(Row < JSIZE && Col < LSIZE)
        d_P[Row*LSIZE+Col] = Pvalue;
}

//Set up and launch of kernal function
void MatrixMultiplication(float *M, float *N, float *P, int j, int k, int l)
{
    int mMatSize = (j*k)*sizeof(float);
    int nMatSize = (k*l)*sizeof(float);
    int pMatSize = (j*l)*sizeof(float);
    float *d_M;
    float *d_N;
    float *d_P;

    cudaMalloc((void**) &d_M, mMatSize);
    cudaMalloc((void**) &d_N, nMatSize);
    cudaMalloc((void**) &d_P, pMatSize);

    cudaMemcpy(d_M, M, mMatSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_N, N, nMatSize, cudaMemcpyHostToDevice);

    // execution configuration
    dim3 dimGrid((l/TILE_WIDTH) + 1, (j/TILE_WIDTH) + 1);
    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);
    
    // launch the kernels
    MatrixMulKernel<<<dimGrid, dimBlock>>>(d_M, d_N, d_P, j, k, l);

    cudaMemcpy(P, d_P, pMatSize, cudaMemcpyDeviceToHost);

    cudaFree(d_M);
    cudaFree(d_N);
    cudaFree(d_P);
}

//Print the values in a matrix
void print(float * matrix, int size, int col, const char * name)
{
    printf("%s:\n", name);
    for (int i = 0; i < size; i++) 
    {
        if((i % col) == 0)
            printf("\n");
        printf(" %10.2f", matrix[i]);
    }
    printf("\n");
}

//main function to get users input and to launch kernal
int main(int argc, char** argv) 
{
    int j,k,l;
    int mSize, nSize, pSize;
    float *M;
    float *N;
    float *P;

    printf("Enter rows(j) for matrix m: ");
    scanf("%d", &j);

    printf("Enter columns(k) for matrix m and rows(k) for matrix n: ");
    scanf("%d", &k);

    printf("Enter columns(l) for matrix n: ");
    scanf("%d", &l);

    //get size of each matrix
    mSize = j * k;
    nSize = k * l; 
    pSize = j * l;

    //allocate in memory
    M = (float *) malloc(mSize*sizeof(float));
    N = (float *) malloc(nSize*sizeof(float));
    P = (float *) malloc(pSize*sizeof(float));

    //assign values to each matrix
    for (int i = 0; i < mSize; i++)
        M[i] = i;

    for (int i = 0; i < nSize; i++)
        N[i] = i+1;

    for (int i = 0; i < pSize; i++)
        P[i] = 0;

    MatrixMultiplication(M, N, P, j, k, l);

    print(M, mSize,k, "M");
    print(N, nSize,l, "N");
    print(P, pSize,l, "P");

    free(M);
    free(N);
    free(P);
}


