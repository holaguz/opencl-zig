__kernel void square_i32(__global int *Input, __global int* Output) {
 
    // Get the index of the current element to be processed
    int i = get_global_id(0);
 
    // Do the operation
    Output[i] = Input[i] * Input[i];
}

__kernel void vecmul_f32(__global float *a, __global float* b, __global float* out, uint n, int offset) {

    int i = get_global_id(0);
    int j = i + offset;

    while (j >= (int)n) {
        j -= n;
    }

    while (j < 0) {
        j += n;
    }

    out[i] = a[i] * b[j];
}

// vim: ft=c
