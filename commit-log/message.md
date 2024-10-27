# Pull Request Title:
Parallelized Convolution, ReLU, and Output Layers via GPU Acceleration with CUDA

# Summary:
Improved performance of a convolutional neural network (CNN). Utilized the GPU and CUDA to parallelize computations for the Convolution, ReLU, and Output Layers to increase performance compared to the current CPU solution. Three kernel calls were implemented corresponding to the above layer computations to implement the changes.

# Tech Details: (3-5 paragraph)

## Kernel:
The convolution kernel has 10 blocks, each containing 20x20 threads. These threads each work on a 5x5 section of the input array in parallel, computing their dot product, and saving the output to a 10x20x20 array. These computations are done using the x and y index of the thread to identify each grouping, as well as the block x index to identify the corresponding filter to apply.

The relu layer kernel splits the work for 4000 elements (10x20x20) with 256 threads per block, evenly across multiple blocks. It then gets the unique thread id corresponding to each element, checks its within index bounds, and then sets the value to a min of 0, operating simultaneously across all threads. 

The output layer kernel has 10 blocks of with 256 threads each. The threads then split the work, and each computes 20 dot products per thread, saving to a shared 10x200 array. The threads are then used to sum these partial dot products in parallel, gathering the dot product for the 10 neurons. Finally, once all the threads are complete with this operation, the array used for this internal computation is reset for future uses.

## CUDA Interaction with Kernel:

The CUDA interactions to the kernel are standard, with the primary unique interactions listed above for gridsize and number of threads. To interact with the kernel, a module is formed from the Parallel Thread Execution (PTX) file containing the kernel code. This module is locally instantiated, along with a local stream, following RustCUDA documentation recommendations. Additional specifics can be seen directly from the L22 of the coursenotes. 

A device buffer is then formed to store data interacting with the kernel. The unsafe block is used as the launch macro is unsafe. It is used to call the kernel with these parameters. The stream waits to be synchronized ensuring all threads complete their task, and any data from the device buffer is copied to corresponding variables.


# Testing (1)

To test the functionality of the CUDA implementation, the CPU implementation was used. The overall results of both were tested for multiple files created using the generate.py and compare.py programs provided. Additionally, the CPU code for each layer's implementation was imported into the CUDA file for testing performance and accuracy of each function. The performance was evaluated on multiple ecetesla computers, with results in corresponding sections. Some custom functions used for testing were left in the code (commented out) for clarity.

# Performance (1)
Tested primarily on `ecetesla0`, the CUDA implementation consistently outperforms the CPU implementation.
Performance of CUDA: Approx 60,000 microseconds
Performance of CPU: Approx 70,000 microseconds
This performance calculation was collected from the provided duration output, which is the time spent doing "actual work" and does not include the overhead for initializing the GPU. It is significant to note that the runtimes of these programs varied between runs depending on when the functions were called throughout the week leading to the submission date, as well as some scattered outliers that were excluded during the multiple runs of data collection.

# Future Improvements:

There are some time improvements that can be made from a few sections throughout the program. The first one to highlight is the relu_layer kernel. The code simply ensures the outputs from the conv layer in the previous call are not negative. This can be directly integrated into the conv layer for minor improvements (as each kernel call takes time) but was left separate for clarity. Additionally, the output_layer (fast) function can be improved by further splitting up the summation of the partial dot products across the threads. However, as the runtime of the CUDA program is faster than the CPU program and performance is no longer being evaluated, this was left as a future task.
