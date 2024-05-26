const std = @import("std");
const c = @import("c.zig");
const ocl = @import("opencl.zig");

const Allocator = std.mem.Allocator;
const check_cl_error = ocl.check_cl_error;

pub fn create_cl_mem_obj(comptime T: type, context: c.cl_context, flags: c.cl_mem_flags, items: []T) !c.cl_mem {
    var ret: c.cl_int = undefined;
    const size = items.len * @sizeOf(T);
    const mem_obj: c.cl_mem = c.clCreateBuffer(context, flags, size, items.ptr, &ret);
    try check_cl_error(ret, error.CreateBufferError);
    return mem_obj;
}

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const alloc = gpa.allocator();
    var ret: c.cl_int = undefined;

    // Get a device_id
    const device_id = ocl.enumerate();

    // Load the program source
    const program_source = @embedFile("program.cl");

    // Create an OpenCL context with the device_id and program source
    var ctx = try ocl.OpenCLContext.init(alloc, device_id, program_source);
    defer ctx.deinit();

    // Create a kernel. A kernel is a function defined in the program source.
    const kernel = try ctx.create_kernel("square_i32");

    // Create input and output buffers. Later we'll upload the input buffer to the device, run the
    // kernel, and download the output buffer.
    // 'h' stands for host, 'd' stands for device (i.e. where the kernel runs)
    var h_input: [32]i32 = undefined;
    var h_output: [32]i32 = undefined;

    for (0..h_input.len) |i| {
        h_input[i] = @intCast(i);
    }

    // Create the input and output buffers on the device.
    var d_input = try ctx.create_buffer(h_input.len * @sizeOf(i32), c.CL_MEM_READ_ONLY, null);
    var d_output = try ctx.create_buffer(h_output.len * @sizeOf(i32), c.CL_MEM_WRITE_ONLY, null);

    // Copy the input buffer values to the device
    try ctx.write_buffer(i32, d_input, &h_input);

    // Set the kernel arguments. These arguments map to the arguments of the kernel function, i.e.
    // the first argument of the kernel function maps to the argument number 0.
    try ctx.set_kernel_arg(kernel, 0, &d_input);
    try ctx.set_kernel_arg(kernel, 1, &d_output);

    // How many work items to run in parallel. This is the number of elements in the input buffer.
    const global_item_size: usize = h_output.len;

    // How many work items to run in parallel per work group.
    const local_item_size: usize = 4;

    // Run the kernel
    ret = c.clEnqueueNDRangeKernel(ctx.command_queue, kernel, 1, null, &global_item_size, &local_item_size, 0, null, null);
    try check_cl_error(ret, error.EnqueueKernel);

    // Wait for the command queue to get processed
    ret = c.clFinish(ctx.command_queue);
    try check_cl_error(ret, error.Finish);

    // Copy the output buffer values from the device
    try ctx.read_buffer(i32, d_output, &h_output);

    // Log the results
    for (0..h_output.len, h_output) |idx, out| {
        std.log.info("{d} -> {d}", .{ idx, out });
    }
}
