// sequential matrix multiplication program as a reference

#include <stdio.h>
#include <stdlib.h>

//using namespace std;

void MatrixMultiplication(float *M, float *N, float *P, int jSize, int kSize, int lSize)
{
    for (int i = 0; i < jSize; i++)
        for (int j = 0; j < lSize; j++) {
            P[i*jSize+j] = 0;
            for (int k = 0; k < kSize; k++)
                P[i*jSize+j] += M[i*kSize+k]*N[k*lSize+j];
        }
}

void print(float * matrix, int size, int col, const char * name)
{
    printf("%s:\n", name);
    for (int i = 0; i < size; i++) {
        if((i % col) == 0)
            printf("\n");
        printf(" %15.2f", matrix[i]);
    }
    printf("\n");
}

int main(int argc, char** argv) 
{
    int i, j, k, l;
    int mSize, nSize, pSize;
    float *M;
    float *N;
    float *P;

    printf("Enter number of width for m matrix: ");
    scanf("%d", &j);

    printf("Enter number of height for m matrix and width for n matrix:");
    scanf("%d", &k);

    printf("Enter number of height for n matrix:");
    scanf("%d", &l);

    mSize = j*k;
    nSize = k*l;
    pSize = j*l;

    M = (float *) malloc(mSize*sizeof(float));
    N = (float *) malloc(nSize*sizeof(float));
    P = (float *) malloc(pSize*sizeof(float));

    for (i = 0; i < mSize; i++) {
        M[i] = i;
    }

    for (i = 0; i < nSize; i++){
        N[i] = i+1;
    }

    for (i = 0; i < pSize; i++){
        P[i] = 0.0;
    }

    MatrixMultiplication(M, N, P, j, k, l);

    print(M, mSize, k, "M");
    print(N, nSize, l, "N");
    print(P, pSize, l, "P");

    free(M);
    free(N);
    free(P);
}
