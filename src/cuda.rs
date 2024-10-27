// This is the skeleton for the CUDA implementation

use crate::cnn::*;
use rustacuda::function::BlockSize;
use rustacuda::launch;
use rustacuda::memory::DeviceBox;
use rustacuda::prelude::*;
use std::error::Error;
use std::ffi::CString;

// Fields need to be ordered this way so the DeviceBoxes are
// dropped before the Context. Otherwise the drop will panic.

pub struct CudaContext {
    conv_layer: DeviceBox<ConvLayer>,
    output_layer: DeviceBox<OutputLayer>,
    module: Module,
    stream: Stream,
    _context: Context,
}

impl CudaContext {
    pub fn init(cnn: &Cnn) -> Result<Self, Box<dyn Error>> {
        rustacuda::init(CudaFlags::empty())?;
        let device = Device::get_device(0)?;
        let ctx =
            Context::create_and_push(ContextFlags::MAP_HOST | ContextFlags::SCHED_AUTO, device)?;
        let ptx = CString::new(include_str!("../kernel/kernel.ptx"))?;
        let module = Module::load_from_string(&ptx)?;
        let stream = Stream::new(StreamFlags::DEFAULT, None)?;

        let conv_layer = DeviceBox::new(&cnn.conv_layer)?;
        let output_layer = DeviceBox::new(&cnn.output_layer)?;

        Ok(CudaContext {
            conv_layer,
            output_layer,
            module,
            stream,
            _context: ctx,
        })
    }

    fn convolution_layer(&mut self, input: &InputMatrix, outputs: &mut ConvOutput) {
        let temp_module = &self.module;
        let temp_stream = &self.stream;

        let mut input_device =
            DeviceBox::new(input).expect("Failed to allocate device memory for input");

        let mut outputs_device =
            DeviceBox::new(outputs).expect("Failed to allocate device memory for output");

        let num_blocks = 10;
        let block_size = (20, 20, 1);

        unsafe {
            let result = launch!(temp_module.ConvolutionLayer<<<num_blocks, block_size, 0, temp_stream>>>(
                input_device.as_device_ptr(),
                self.conv_layer.as_device_ptr(),
                outputs_device.as_device_ptr()
            ));
            result;
        }
        // Kernel launches are asynchronous, so we wait for the kernels to finish executing.
        temp_stream.synchronize();
        outputs_device.copy_to(outputs);
    }

    fn relu_layer(&mut self, conv_out: &mut ConvOutput) {
        // Calculate number of elements
        let num_elements = 4000;
        let temp_module = &self.module;
        let temp_stream = &self.stream;

        let mut outputs_device =
            DeviceBox::new(conv_out).expect("Failed to allocate device memory for output");

        // Launch the kernel
        unsafe {
            let block_size = 256;
            let num_blocks = (num_elements + block_size - 1) / block_size;

            let result = launch!(temp_module.relu_layer_kernel<<<num_blocks, block_size, 0, temp_stream>>>(
                outputs_device.as_device_ptr(),
                num_elements as i32
            ));
            result;
        }
        temp_stream.synchronize();
        outputs_device.copy_to(conv_out);
    }

    // Old output_layer function, left for clarity
    fn output_layer(&mut self, input: &ConvOutput, output: &mut OutputVec) {
        let grid_size = 10;
        let block_size = 256;
        let temp_module = &self.module;
        let temp_stream = &self.stream;

        let mut input_device =
            DeviceBox::new(input).expect("Failed to allocate device memory for input");
        let mut output_device =
            DeviceBox::new(output).expect("Failed to allocate device memory for output");

        unsafe {
            let result = launch!(temp_module.output_layer_kernel<<<grid_size, block_size, 0, temp_stream>>>(
                input_device.as_device_ptr(),
                self.output_layer.as_device_ptr(),
                output_device.as_device_ptr()
            ));
            result;
        }

        temp_stream.synchronize();

        output_device.copy_to(output);
    }

    // Updated output_layer function
    fn output_layer_fast(&mut self, input: &ConvOutput, output: &mut OutputVec) {
        let temp_module = &self.module;
        let temp_stream = &self.stream;

        let mut input_device =
            DeviceBox::new(input).expect("Failed to allocate device memory for input");
        let mut output_device =
            DeviceBox::new(output).expect("Failed to allocate device memory for output");

        let grid_size = 10;
        let block_size = 256;

        unsafe {
            let result = launch!(temp_module.output_layer_fast<<<grid_size, block_size, 0, temp_stream>>>(
                input_device.as_device_ptr(),
                self.output_layer.as_device_ptr(),
                output_device.as_device_ptr()
            ));
            result;
        }

        temp_stream.synchronize();
        output_device.copy_to(output);
    }

    pub fn compute(&mut self, input: &InputMatrix, cnn: &Cnn) -> Result<OutputVec, Box<dyn Error>> {
        let mut conv_output = ConvOutput([[[0.0; CONV_OUT_DIM]; CONV_OUT_DIM]; CONV_LAYER_SIZE]);
        let mut output = OutputVec([0.0; OUT_LAYER_SIZE]);

        self.convolution_layer(&input, &mut conv_output);
        // convolution_layer(&input, &cnn.conv_layer, &mut conv_output); // CPU for testing

        self.relu_layer(&mut conv_output);
        // relu_layer(&mut conv_output); // CPU for testing

        self.output_layer_fast(&mut conv_output, &mut output);
        // self.output_layer(&mut conv_output, &mut output); // old output layer
        // output_layer(&conv_output, &cnn.output_layer, &mut output); // CPU for testing

        Ok(output)
    }
}

/////////////////////////////////////////.
fn print_first_conv(outputs: &ConvOutput) {
    let first_filter_output = &outputs.0[0]; // Get the first filter output
    let first_row = &first_filter_output[0]; // Get the first row of the first filter output
    for element in &first_row[..1] {
        println!("{}", element);
    }
}

fn print_last_conv(outputs: &ConvOutput) {
    // Get the last filter output
    if let Some(last_filter_output) = outputs.0.last() {
        // Get the last row of the last filter output
        if let Some(last_row) = last_filter_output.last() {
            // Get the last element of the last row
            if let Some(last_element) = last_row.last() {
                println!("{}", last_element);
            }
        }
    }
}

fn print_first_out(output: &OutputVec) {
    if let Some(first_elem) = output.0.first() {
        println!("{}", first_elem);
    } else {
        println!("OutputVec is empty");
    }
}
fn print_last_out(output: &OutputVec) {
    if let Some(last_elem) = output.0.last() {
        println!("{}", last_elem);
    } else {
        println!("OutputVec is empty");
    }
}

fn print_non_zero_count(output: &ConvOutput) {
    let mut count = 0;

    for layer in &output.0 {
        for row in layer {
            for &elem in row {
                if elem != 0.0 {
                    count += 1;
                }
            }
        }
    }

    println!("Number of non-zero elements: {}", count);
}

fn convolution_layer(input: &InputMatrix, conv_filters: &ConvLayer, outputs: &mut ConvOutput) {
    // Go through each convolution neuron
    for (filter, out) in conv_filters.0.iter().zip(outputs.0.iter_mut()) {
        // Divide the 100x100 input matrix into 5x5 regions. There are 20x20 such regions in the
        // matrix. Each convolution neuron does a dot product of its filter with each region, producing a
        // 20x20 output matrix of products

        // for i = 0 -> 100, i+=5
        for i in (0..INPUT_DIM).step_by(FILTER_DIM) {
            // for j = 0 -> 100, j+=5
            for j in (0..INPUT_DIM).step_by(FILTER_DIM) {
                // Dot product
                let prod: f64 = (0..FILTER_DIM)
                    .flat_map(move |x| {
                        (0..FILTER_DIM).map(move |y| input.0[i + x][j + y] * filter[x][y])
                    })
                    .sum();

                // save result to the 20x20 region
                out[i / FILTER_DIM][j / FILTER_DIM] = prod;
            }
        }
    }
}

fn relu_layer(conv_out: &mut ConvOutput) {
    // Any value below 0 in the previous layer's output is changed to 0
    for matrix in conv_out.0.iter_mut() {
        for row in matrix {
            for val in row {
                if *val < 0.0 {
                    *val = 0.0;
                }
            }
        }
    }
}

fn output_layer(input: &ConvOutput, weights: &OutputLayer, output: &mut OutputVec) {
    // Go thru each output neuron
    for (weight, out) in weights.0.iter().zip(output.0.iter_mut()) {
        // Flatten the output of the previous layer into a 4000x1 vector, then dot product it with
        // the weight vector to produce a single value
        let flattened = input.0.iter().flat_map(|n| n.iter().flat_map(|r| r.iter()));
        let prod: f64 = flattened.zip(weight.iter()).map(|(a, b)| a * b).sum();
        *out = prod;
    }
}
