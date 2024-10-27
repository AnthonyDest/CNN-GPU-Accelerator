// Very minimal skeleton for the kernel

#include <stdio.h>


#define INPUT_DIM 100
#define FILTER_DIM 5
#define CONV_OUT_DIM 20
#define GRID_SIZE 100
#define NUM_FILTERS 10
#define FILTER_SIZE 5

extern "C" __global__ void ConvolutionLayer(const double input_data[GRID_SIZE][GRID_SIZE], const double filters[NUM_FILTERS][FILTER_SIZE][FILTER_SIZE], double output_data[10][20][20]) {
    int filter_index = blockIdx.x; // Index of the convolution filter

    int row = threadIdx.y; 
    int col = threadIdx.x; 

    for (int row_i = 0; row_i <5; row_i ++){
        for(int col_j = 0; col_j <5; col_j ++){
            output_data[filter_index][row][col] += input_data[row*5 + row_i][col*5 + col_j] * filters[filter_index][row_i][col_j];
        }
    }
}


extern "C" __global__ void relu_layer_kernel(double* conv_out, int num_elements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < num_elements) {
        if (conv_out[idx] < 0.0f) {
            conv_out[idx] = 0.0f;
        }
    }
}

extern "C" __global__ void output_layer_kernel(const double *flat_input, const double weights[10][4000], double* output) {
    int neuron_offset = blockIdx.x;
        double sum = 0.0;
        for (int i = 0; i < 4000; i++) {
            sum += flat_input[i] * weights[neuron_offset][i];
        }
        output[neuron_offset] = sum;
}

extern "C" __global__ void output_layer_fast(const double input_A[4000], const double input_B[10][4000],  double output[10]) {   
    int neuron_idx = blockIdx.x; // 10
    int thread_idx = threadIdx.x; // 200

    __shared__ double first_array[10][200]; // Declaring first_array as shared memory

    // compute 20 dot products per thread
    for (int i = 0; i < 20; i++) {
        int input_idx = thread_idx * 20 + i;

        if (input_idx < 4000){
            first_array[neuron_idx][thread_idx] += input_A[input_idx] * input_B[neuron_idx][input_idx];
        }
    }   

    // waits for above to finish as kernel recognizes shared threads,

    // sum the partial dot products
    double sum = 0.0;
    for (int i = 0; i < 200; i++) {
        sum += first_array[neuron_idx][i];
    }

    output[neuron_idx] = sum;

    __syncthreads();
    // Resetting shared memory to zero
    for (int i = 0; i < 10; i++) {
        for (int j = 0; j < 200; j++) {
            first_array[i][j] = 0.0;
        }
    }
}
