/*********************************************************************************
 * Name:        Eric Blasko
 * Class:       CSE525
 * Assignment:  Homework #2
 * Date:        May 1, 2019
 * Desc:        This program utilizes parallelism to find the summation of given
 *              values. The implementation uses a user defined reduce algorithm
 *              to minic the MPI_reduce function. Each processor is assigned its own
 *              id which is also set as its local value. Each processor communicates
 *              with its neighbor by passing its local value, which when received
 *              is added to that processors sum. Once all processors have finished
 *              its communication, the final sum is passed to the root processor
 *              to display the value.
 **********************************************************************************/

#include "mpi.h"
#include <stdio.h>
#include <stdlib.h>
#include "math.h"

void reduce(int*,int*, int, MPI_Comm);

int main(int argc, char **argv)
{
    int local = 0; 
    int root, id, p;
    int sum = 0; 
    
    MPI_Init(&argc,&argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &id);
    MPI_Comm_size(MPI_COMM_WORLD, &p);
   
    //check for 2 arg, second arg will be root
    if(argc != 2)
    {
        if(!id)
	{
            printf("enter 2 argu");
	    fflush(stdout);
	}        
        MPI_Finalize();
        exit(1);
    }

    root = atoi(argv[1]);
 
    //if root is greater than processors - exit
    if(root >= p)
    {
        if(!id)
	{	
            printf("root is greater than processors\n");
	    fflush(stdout);
	}
        MPI_Finalize();
        exit(1);
    }

    reduce(&local,&sum,root,MPI_COMM_WORLD);
    
    MPI_Finalize();
}

//each processor will assign its local and sum value to its id. As each dimension is iterated,
//each will find who is its partner to exchange with. exchange will only happen if the partner
//is less than active processes. if the id is less than half of active processes, it will 
//recieve, else it will send its current sum. After processor 0 will send the sum to the root
//were it will be displayed.
void reduce(int *local,int *sum, int root, MPI_Comm comm)
{
    int p, id, source, dest, tag=1;
    int message;
    MPI_Comm_rank(comm, &id);
    MPI_Comm_size(comm, &p);
    MPI_Status status;

    *local = id;
    *sum = *local;

    printf("proc: %d - local value: %d\n",id,*local);
    fflush(stdout);

    for(int dim = ceil(log2(p)) - 1; dim >= 0; dim--)
    {
        int partner = id ^ (int)(pow(2,dim));
        if(partner < p && id < p)
        {
            if(id < (int)pow(2,dim))
            {
                source = partner;
                MPI_Recv(&message,1,MPI_INT,source,tag,MPI_COMM_WORLD,&status);
                *sum += message;
                }
            else
            {
                message = *sum;
                dest = partner;
                MPI_Send(&message,1,MPI_INT,dest,tag,MPI_COMM_WORLD);
            }    
        }
        p = pow(2,dim);
    }

    if(id == 0)
    {
        message = *sum;
        MPI_Send(&message,1,MPI_INT,root,tag,MPI_COMM_WORLD);
    }
    if(id == root)
    {
        MPI_Recv(&message,1,MPI_INT,0,tag,MPI_COMM_WORLD,&status);
        *sum = message;
        printf("\n*******************************\n");
        printf("*     Reduction Complete      *\n");
        printf("* Process %d has the sum of %d *\n",id,*sum);
        printf("*******************************\n");
        fflush(stdout);
    }
}

