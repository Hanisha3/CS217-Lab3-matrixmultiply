#include <stdio.h>

#define TILE_SIZE 16

__global__ void mysgemm(int m, int n, int k, const float *A, const float *B, float* C) {

    /********************************************************************
     *
     * Compute C = A x B
     *   where A is a (m x k) matrix
     *   where B is a (k x n) matrix
     *   where C is a (m x n) matrix
     *
     * Use shared memory for tiling
     *
     ********************************************************************/

    /*************************************************************************/
    // INSERT KERNEL CODE HERE

    /*************************************************************************/
    __shared__ float ds_A[TILE_SIZE][TILE_SIZE];
    __shared__ float ds_B[TILE_SIZE][TILE_SIZE];

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int Row = by * TILE_SIZE + ty;
    int Col = bx * TILE_SIZE + tx;
    float Pvalue = 0;

    for(int i = 0; i < ((k-1) / TILE_SIZE) + 1; i++) {
        if(Row < m && i * TILE_SIZE + tx < k)
                ds_A[ty][tx] = A[Row * k + i * TILE_SIZE + tx];
        else
                ds_A[ty][tx] = 0.0;
        if((i * TILE_SIZE + ty) < k && Col < n)
                ds_B[ty][tx] = B[Col + (i * TILE_SIZE + ty) * n];
        else
                ds_B[ty][tx] = 0.0;
        __syncthreads();
        if(Row < m && Col < n) {
                for(int k = 0; k < TILE_SIZE; ++k)
                        Pvalue += ds_A[ty][k] * ds_B[k][tx];
        }
        __syncthreads();

        }
        if(Row < m && Col < n)
                C[Row * n  + Col] = Pvalue;
}
void basicSgemm(int m, int n, int k, const float *A, const float *B, float *C)
{
    // Initialize thread block and kernel grid dimensions ---------------------

    const unsigned int BLOCK_SIZE = TILE_SIZE;
    const int Width = 1024;

    /*************************************************************************/
    //INSERT CODE HERE

    /*************************************************************************/

    // Invoke CUDA kernel -----------------------------------------------------

    /*************************************************************************/
    //INSERT CODE HERE

    /*************************************************************************/
    dim3 dim_blk(BLOCK_SIZE, BLOCK_SIZE, 1);
    dim3 dim_grid((Width / BLOCK_SIZE), (Width / BLOCK_SIZE), 1);

    mysgemm<<<dim_grid, dim_blk>>>(m, n, k, A, B, C);


}
