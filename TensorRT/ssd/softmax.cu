__global__ void kernelSoftmax(float *x, int channels, float *y)
{
  extern __shared__ float mem[];
  __shared__ float sum_value;
  float number = *(x + blockdim.x * blockIdx.x + threadIdx.x);
  float number_exp = __expf(number);
  atomicAdd(&sum_value, number_exp);
  __syncthreads();
  y[blockDim.x * blockIdx.x + threadIdx.x] = __fdiv_rd(number_exp, sum_value);
}
