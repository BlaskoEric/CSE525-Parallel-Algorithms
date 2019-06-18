/*********************************************************************************
 * Name:        Eric Blasko
 * Class:       CSE525
 * Assignment:  Midterm
 * Date:        May 18th, 2019
 * Desc:        This program utilizes parallelism to find the summation of given
 *              values on different processors. The implementation uses a user 
 *              defined all reduce algorithm to minic the MPI_allreduce function.
 *              Each processor is assigned its own id and a random value from 0-
 *              50. Each processor communicates with its neighbor by swapping 
 *              local values, which when received is added to both processors 
 *              sums. Once all processors have finished its communication, each 
 *              processor displays its sum. If the number of processors is a 
 *              power of 2, each processor will have the same sum. If not a power
 *              of 2, some processors will not have the total sum.
 **********************************************************************************/

#include "mpi.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

void allreduce(int*,int*, MPI_Comm);
int Pow2(int);
int Log2(int);
int GetPartner(int,int);

int main(int argc, char **argv)
{
    int local,sum;
    
    MPI_Init(&argc,&argv);
    
    allreduce(&local,&sum,MPI_COMM_WORLD);
    
    MPI_Finalize();
}

//each processor is assigned a random value from 0-50. As each dimension is 
//iterated, each will find who is its partner to exchange with. exchange will only
//happen if the partner is less than active processes. Both processors will pass 
//its current sum to its partner and add the value to its sum. When all processors 
//are complete, each will display its final sum
void allreduce(int *local,int *sum, MPI_Comm comm)
{
    int p, id, tag=1;
    int outMsg,inMsg;
    MPI_Comm_rank(comm, &id);
    MPI_Comm_size(comm, &p);
    MPI_Status status;

    //generate random value
    srand(time(NULL) + id);
    *local = *sum = rand() % 50;

    int partnerLocal;

    //print local values
    printf("proc: %d - local value: %d\n",id,*local);
    fflush(stdout);

    //if power of 2, reduce dim by 1
    int dim = Log2(p);
    if(p == Pow2(dim))
        dim -= 1;

    //loop through each dimension
    for(dim; dim >= 0; dim--)
    {
        int partner = GetPartner(id,dim);
        if(partner < p)
        {
            outMsg = *sum;

            if(id < Pow2(dim))
            {
                MPI_Recv(&inMsg,1,MPI_INT,partner,tag,MPI_COMM_WORLD,&status);
                MPI_Send(&outMsg,1,MPI_INT,partner,tag,MPI_COMM_WORLD);
                }
            else
            {
                MPI_Send(&outMsg,1,MPI_INT,partner,tag,MPI_COMM_WORLD);
                MPI_Recv(&inMsg,1,MPI_INT,partner,tag,MPI_COMM_WORLD,&status);
            }   
                
            *sum += inMsg; 
        }
    }
    //print final values
    printf("Processor: %d: sum = %d\n",id,*sum);
    fflush(stdout);
}

//recursive function to return power of 2
int Pow2(int pow)
{
    if(pow <= 0) return 1;
    return 2 * Pow2(pow-1);
}

//recursive function to find log base 2 of value
int Log2(int n)
{
    if(n <= 1) return 0;
    return 1 + Log2(n/2);
}

//gets current partner from dimmension and id. XOR function
int GetPartner(int id,int dim)
{
    return id ^ Pow2(dim);
}

