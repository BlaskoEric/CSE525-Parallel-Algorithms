/**********************************************************************
 * Name:    Eric Blasko
 * Date:    06/12/19
 * Final
 * reduction.cu
 * This program performs reduction using CUDA and supports mulitple 
 * block reduction. Multiple kernal calls may be needed based
 * on the number of blocks. Each block will store its partial sum in 
 * an array, which will reduce the overall size that needs to be worked
 * on each iteration. Once the reduction is complete, the final sum
 * of all values will be displayed from array[0]. 
 *
 * Reduction will calculate Sum of Integers from 1 to 1024 = 524,800 
 **********************************************************************/

#include <stdio.h>

#define SIZE 1024

/**********************************************************************
 * Kernal function that performs reduction. Only thread index's that 
 * are less than the active threads count will enter the first if 
 * statement. Each active thread will load is value into shared memory
 * then sync with other threads. Each thread will perform the reduction
 * until only the first index has the partial sum. The partial sum
 * will then be stored in the d_array based on the Block index number.
 * This makes sure that all partial sums are to the front of the array.
 **********************************************************************/
__global__ void reduction(float * d_array, int ARRAYSIZE)
{
    __shared__ float partialSum[SIZE];

    int t = threadIdx.x;
    if(t < ARRAYSIZE)
    {
        partialSum[t] = d_array[blockIdx.x * blockDim.x + t];
    
        for(int stride = (ARRAYSIZE / 2); stride >= 1; stride = stride>>1)
        {
            __syncthreads();

            if(t < stride)
            { 
                partialSum[t] += partialSum[t + stride];
            }
        }
   
        //Only first index save value. Saved based on block index into array
        if(t == 0)
        {  
            d_array[blockIdx.x] = partialSum[t];
        }
    }
}


/**********************************************************************
 * This function allocates the device data and copies data to GPU. 
 * If the block size is greater that one, then multiple kernel calls
 * will be needed to combine results from each block. Each kernel call 
 * will reduce the block number by ceiling of blocks/threads. End
 * result will be stored and printed from location 0 of h_array
 **********************************************************************/
__host__ void startReduction(float *h_array, int blocks, int threads)
{
    float * d_array;
    float arraySize = (blocks*threads)*sizeof(float);

    cudaMalloc((void **) &d_array,arraySize);

    cudaMemcpy(d_array,h_array,arraySize,cudaMemcpyHostToDevice);
    
    //first call to reduction. Is always called
    reduction<<<blocks,threads>>>(d_array,blocks*threads);

    cudaMemcpy(h_array,d_array,arraySize,cudaMemcpyDeviceToHost);
   
    //if there is more than 1 block, keep calling reduction till
    //all values are within a single block
    int newArraySize = blocks;
    while(blocks > 1)
    {
        /*
        * Example 256 blocks with 4 threads will have saved partial
        * sum in first 256 locations of d_array. Next iteration will
        * run 64 blocks with 4 threads each = 256. will loop till
        * blocks = 1 and threads = 4.
        */
        blocks = ((blocks-1)/threads)+1;
       
        cudaMemcpy(d_array,h_array,arraySize,cudaMemcpyHostToDevice);
        
        reduction<<<blocks,threads>>>(d_array,newArraySize);
        
        cudaMemcpy(h_array,d_array,arraySize,cudaMemcpyDeviceToHost);
        
        newArraySize = blocks;
   } 

    printf("Result from Reduction\n");
    printf("Sum of 1-1024 = %10.2f\n\n",h_array[0]);

    cudaFree(d_array);     
}

/**********************************************************************
 * checks if value is power of 2. Used in main
 **********************************************************************/
bool power2(int value)
{
  if (value == 0) 
    return 0; 
  while (value != 1) 
  { 
      if (value%2 != 0) 
         return 0; 
      value = value/2; 
  } 
  return 1; 
} 

/********************************************************************** 
 * Main function will get users input for block size. Threads per block
 * are based off of 1024 / block size. Blocks must be a power of two.
 * Uses error checking if block input is invalid.
 **********************************************************************/
int main(int argc, char** argv)
{
    float *h_array;
    int blocks = 0;
    int threads = 0;
    int inputSize = 0;

    printf("\nEnter the array size (max 1024): ");
    scanf("%d",&inputSize);

    blocks = (inputSize-1)
    threads = SIZE / blocks;
    printf("Using %d block of %d threads\n",blocks,threads);    

    //error detection 
    while(blocks > 512 || !power2(blocks))
    {
        printf("Max limit block limit is 512, you entered %d\n",blocks);
        printf("Enter the number of blocks (power of 2): ");
        scanf("%d",&blocks);
        threads = SIZE / blocks;
    }

    h_array = (float *) malloc(blocks*threads*sizeof(float));

    for(int i = 0; i < blocks*threads; i++)
        h_array[i] = i+1;

    startReduction(h_array,blocks,threads);

    free(h_array);

    return 0;
}
       

