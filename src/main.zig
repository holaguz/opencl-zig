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

    const device_id = ocl.enumerate();
    const program_source = @embedFile("program.cl");
    var ctx = try ocl.OpenCLContext.init(alloc, device_id, program_source);

    const kernel = try ctx.create_kernel("square_i32");

    var h_input: [32]i32 = undefined;
    var h_output: [32]i32 = undefined;

    for (0..h_input.len) |i| {
        h_input[i] = @intCast(i);
    }

    const d_input: c.cl_mem = try create_cl_mem_obj(i32, ctx.context, c.CL_MEM_READ_ONLY | c.CL_MEM_COPY_HOST_PTR, &h_input);
    defer _ = c.clReleaseMemObject(d_input);

    const d_output: c.cl_mem = try create_cl_mem_obj(i32, ctx.context, c.CL_MEM_WRITE_ONLY | c.CL_MEM_USE_HOST_PTR, &h_output);
    defer _ = c.clReleaseMemObject(d_output);

    // ret = c.clEnqueueWriteBuffer(command_queue, d_input, c.CL_TRUE, 0, @sizeOf(i32) * h_input.len, &h_input, 0, null, null);
    // try check_ocl_error(ret, error.Enqueue);

    ret = c.clSetKernelArg(kernel, 0, @sizeOf(c.cl_mem), @ptrCast(&d_input));
    try check_cl_error(ret, error.SetKernelArgError);
    ret = c.clSetKernelArg(kernel, 1, @sizeOf(c.cl_mem), @ptrCast(&d_output));
    try check_cl_error(ret, error.SetKernelArgError);

    const global_item_size: usize = h_output.len; // Process the entire lists
    const local_item_size: usize = 4; // Divide work items into groups of 4

    ret = c.clEnqueueNDRangeKernel(ctx.command_queue, kernel, 1, null, &global_item_size, &local_item_size, 0, null, null);
    try check_cl_error(ret, error.EnqueueKernel);

    ret = c.clFinish(ctx.command_queue); // Wait for the command queue to get processed
    try check_cl_error(ret, error.Finish);

    ret = c.clEnqueueReadBuffer(ctx.command_queue, d_output, c.CL_TRUE, 0, @sizeOf(i32) * h_output.len, &h_output, 0, null, null);
    try check_cl_error(ret, error.EnqueueRead);

    for (0..h_output.len, h_output) |idx, out| {
        std.log.info("{d} -> {d}", .{ idx, out });
    }
}
