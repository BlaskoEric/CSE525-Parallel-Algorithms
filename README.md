# CSE525-Parallel-Algorithms
CSE525 Parallel Algorithms
With Dr. Kay Zehmoudeh

# Homework

### Homework 1
* **Routing using MPI** - Routing algorithm for a 2D-mesh network with wrap around edges. The mesh consists of n rows and n edges (square mesh). The rows and columns are numbered 0 to n-1

### Homework 2
* **Reduction using MPI** - This program utilizes parallelism to find the summation of given
values. The implementation uses a user defined reduce algorithm
to minic the MPI_reduce function. Each processor is assigned its own
id which is also set as its local value. Each processor communicates
with its neighbor by passing its local value, which when received
is added to that processors sum. Once all processors have finished
its communication, the final sum is passed to the root processor
to display the value.

### Homework 3
* **Questions CUDA** - Homework solutiongs to various problems using CUDA

### Homework 4
* **Matrix Multiplication using CUDA** - This program performs rectangle matrix multiplication, which uses shared mem
of size TILE_WIDTH x TILE_WIDTH. Values of Matrix M and N are chosen by the
user such that M is of size JxK and N is of size KxL. The kernal function 
produces the results of the matrix multiplication between M and N, storing
it in matrix P which is of size JxL.  

### Midterm
* **All Reduce using MPI** - This program utilizes parallelism to find the summation of given
values on different processors. The implementation uses a user 
defined all reduce algorithm to minic the MPI_allreduce function.
Each processor is assigned its own id and a random value from 0-
50. Each processor communicates with its neighbor by swapping 
local values, which when received is added to both processors 
sums. Once all processors have finished its communication, each 
processor displays its sum. If the number of processors is a 
power of 2, each processor will have the same sum. If not a power
of 2, some processors will not have the total sum.

### Final
* **Reduction using CUDA** - This program performs reduction using CUDA and supports mulitple 
block reduction. Multiple kernal calls may be needed based
on the number of blocks. Each block will store its partial sum in 
an array, which will reduce the overall size that needs to be worked
on each iteration. Once the reduction is complete, the final sum
of all values will be displayed from array[0]. 
Reduction will calculate Sum of Integers from 1 to 1024 = 524,800 

### Project
* **Isomorphism** - Implementation of of Min-Young-Son, Young-Hak Kim, Byoung-Woo Oh efficient parallel algorithm for graph isomorphism on GPU using CUDA. The original implementation includes a combination of CPU/GPU algorithms for checking graphs for isomorphism. I exanded on there algorithm to use a GPU/GPU method which resulted in a 25% speed up on graphs larger than 5000 nodes. The original article can be found at

https://www.researchgate.net/publication/287222374_An_Efficient_Parallel_Algorithm_for_Graph_Isomorphism_on_GPU_using_CUDA
