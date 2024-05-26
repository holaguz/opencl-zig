__kernel void square_i32(__global int *Input, __global int* Output) {
 
    // Get the index of the current element to be processed
    int i = get_global_id(0);
 
    // Do the operation
    Output[i] = Input[i] * Input[i];
}

// vim: ft=c
