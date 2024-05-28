__kernel void square_i32(__global int *Input, __global int* Output) {
 
    // Get the index of the current element to be processed
    int i = get_global_id(0);
 
    // Do the operation
    Output[i] = Input[i] * Input[i];
}

__kernel void vecmul_i32(__global int *a, __global int* b, __global int* out, const uint n, const int offset) {

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
