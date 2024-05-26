const std = @import("std");
const c = @import("c.zig");

const Allocator = std.mem.Allocator;

// ?*const fn ([*c]const u8, ?*const anyopaque, usize, ?*anyopaque) callconv(.C) void
// void (CL_CALLBACK* pfn_notify)(const char* errinfo, const void* private_info, size_t cb, void* user_data),
pub fn cl_ctx_callback(error_info: [*c]const u8, private_info: ?*const anyopaque, cb: usize, user_data: ?*anyopaque) callconv(.C) void {
    _ = user_data; // autofix
    _ = cb; // autofix
    _ = private_info; // autofix

    std.log.info("cl_ctx_callback: {s}", .{error_info});
}

pub fn create_cl_mem_obj(comptime T: type, context: c.cl_context, flags: c.cl_mem_flags, items: []T) !c.cl_mem {
    var ret: c.cl_int = undefined;
    const size = items.len * @sizeOf(T);
    const mem_obj: c.cl_mem = c.clCreateBuffer(context, flags, size, items.ptr, &ret);
    try check_cl_error(ret, error.CreateBufferError);
    return mem_obj;
}

pub fn check_cl_error(ret: c.cl_int, err: ?anyerror) !void {
    if (ret != c.CL_SUCCESS) {
        std.log.err("Error: {d}", .{ret});
        if (err != null) {
            return err.?;
        } else {
            return error.GenericError;
        }
    }
}

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const alloc = gpa.allocator();
    _ = alloc;

    var platform_id: c.cl_platform_id = undefined;
    var device_id: c.cl_device_id = undefined;

    var ret_num_devices: c.cl_uint = undefined;
    var ret_num_platforms: c.cl_uint = undefined;
    var ret = c.clGetPlatformIDs(1, &platform_id, &ret_num_platforms);
    ret = c.clGetDeviceIDs(platform_id, c.CL_DEVICE_TYPE_DEFAULT, 1, &device_id, &ret_num_devices);

    std.log.debug("Found {d} platform(s) and {d} device(s)", .{ ret_num_platforms, ret_num_devices });

    var platform_vendor: [64]u8 = undefined;
    var platform_name: [64]u8 = undefined;
    _ = c.clGetPlatformInfo(platform_id, c.CL_PLATFORM_NAME, 64, &platform_name, null);
    _ = c.clGetPlatformInfo(platform_id, c.CL_PLATFORM_VENDOR, 64, &platform_vendor, null);

    std.log.debug("Platform:", .{});
    std.log.debug(" - Name:   {s}", .{platform_name});
    std.log.debug(" - Vendor: {s}", .{platform_vendor});

    var device_name: [64]u8 = undefined;
    var device_vendor: [64]u8 = undefined;
    _ = c.clGetDeviceInfo(device_id, c.CL_DEVICE_NAME, 64, &device_name, null);
    _ = c.clGetDeviceInfo(device_id, c.CL_DEVICE_VENDOR, 64, &device_vendor, null);

    std.log.debug("Device:", .{});
    std.log.debug(" - Name:   {s}", .{device_name});
    std.log.debug(" - Vendor: {s}", .{device_vendor});

    const context = c.clCreateContext(0, 1, &device_id, cl_ctx_callback, null, &ret);
    try check_cl_error(ret, error.CreateContextError);
    defer _ = c.clReleaseContext(context);

    const command_queue = c.clCreateCommandQueueWithProperties(context, device_id, 0, &ret);
    try check_cl_error(ret, error.CreateCommandQueueError);
    defer _ = c.clReleaseCommandQueue(command_queue);

    const program_source: [*c]const u8 = @embedFile("program.cl");
    const stupid_ptr = @constCast(&program_source);
    const program = c.clCreateProgramWithSource(context, 1, stupid_ptr, null, &ret);
    try check_cl_error(ret, error.CreateProgramError);

    defer _ = c.clReleaseProgram(program);

    const build_opts = "";
    ret = c.clBuildProgram(program, 1, &device_id, build_opts, null, null);
    try check_cl_error(ret, error.BuildProgramError);

    const kernel = c.clCreateKernel(program, "square_i32", &ret);
    try check_cl_error(ret, error.CreateKernelError);
    defer _ = c.clReleaseKernel(kernel);

    var h_input: [128]i32 = undefined;
    var h_output: [128]i32 = undefined;

    for (0..128) |i| {
        h_input[i] = @intCast(i);
    }

    const d_input: c.cl_mem = try create_cl_mem_obj(i32, context, c.CL_MEM_READ_ONLY | c.CL_MEM_COPY_HOST_PTR, &h_input);
    try check_cl_error(ret, error.CreateBufferError);
    defer _ = c.clReleaseMemObject(d_input);

    const d_output: c.cl_mem = try create_cl_mem_obj(i32, context, c.CL_MEM_WRITE_ONLY | c.CL_MEM_USE_HOST_PTR, &h_output);
    try check_cl_error(ret, error.CreateBufferError);
    defer _ = c.clReleaseMemObject(d_output);

    // ret = c.clEnqueueWriteBuffer(command_queue, d_input, c.CL_TRUE, 0, @sizeOf(i32) * h_input.len, &h_input, 0, null, null);
    // try check_ocl_error(ret, error.Enqueue);

    ret = c.clSetKernelArg(kernel, 0, @sizeOf(c.cl_mem), @ptrCast(&d_input));
    try check_cl_error(ret, error.SetKernelArgError);
    ret = c.clSetKernelArg(kernel, 1, @sizeOf(c.cl_mem), @ptrCast(&d_output));
    try check_cl_error(ret, error.SetKernelArgError);

    const global_item_size: usize = h_output.len; // Process the entire lists
    const local_item_size: usize = 4; // Divide work items into groups of 4

    ret = c.clEnqueueNDRangeKernel(command_queue, kernel, 1, null, &global_item_size, &local_item_size, 0, null, null);
    try check_cl_error(ret, error.EnqueueKernel);

    ret = c.clFinish(command_queue); // Wait for the command queue to get processed
    try check_cl_error(ret, error.Finish);

    ret = c.clEnqueueReadBuffer(command_queue, d_output, c.CL_TRUE, 0, @sizeOf(i32) * h_output.len, &h_output, 0, null, null);
    try check_cl_error(ret, error.EnqueueRead);

    for (0..h_output.len, h_output) |idx, out| {
        std.log.info("{d} -> {d}", .{ idx, out });
    }
}
