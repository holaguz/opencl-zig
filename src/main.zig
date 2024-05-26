const std = @import("std");
const c = @import("c.zig");

const Allocator = std.mem.Allocator;

const Context = struct {
    platform_id: c.cl_platform_id,
};

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
    const mem_obj = c.clCreateBuffer(context, flags, size, items.ptr, &ret);
    if (ret != 0) {
        std.log.err("Error creating a buffer: {d}", .{ret});
        return error.CreateBufferError;
    }
    return mem_obj;
}

pub fn parse_source_code(allocator: Allocator, source: []const u8) !std.ArrayListAligned(u8, 8) {
    const sep = [1]u8{0};
    var iter = std.mem.splitSequence(u8, source, &sep);

    var parsed = std.ArrayListAligned(u8, 8).init(allocator);
    while (iter.next()) |line| {
        try parsed.appendSlice(line);
        try parsed.append(sep);
    }

    return parsed;
}

pub fn check_error(ret: c.cl_int) !void {
    if (ret != 0) {
        std.log.err("Error: {d}", .{ret});
        return error.GenericError;
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
    if (ret != 0) {
        std.log.err("Failed to create cl_context: {d}", .{ret});
        return error.CreateContextError;
    }

    defer _ = c.clReleaseContext(context);

    const command_queue = c.clCreateCommandQueueWithProperties(context, device_id, 0, &ret);

    if (ret != 0) {
        std.log.err("Failed to create command queue: {d}", .{ret});
        return error.CreateCmdQueueError;
    }

    defer _ = c.clFlush(command_queue);
    defer _ = c.clFinish(command_queue);

    var vec_a: [128]i32 = undefined;
    var vec_b: [128]i32 = undefined;
    var out: [128]i32 = undefined;

    const rflags = c.CL_MEM_USE_HOST_PTR | c.CL_MEM_READ_ONLY;
    const wflags = c.CL_MEM_USE_HOST_PTR | c.CL_MEM_WRITE_ONLY;

    const a_mem_obj = try create_cl_mem_obj(i32, context, rflags, &vec_a);
    defer _ = c.clReleaseMemObject(a_mem_obj);
    const b_mem_obj = try create_cl_mem_obj(i32, context, rflags, &vec_b);
    defer _ = c.clReleaseMemObject(b_mem_obj);
    const out_mem_obj = try create_cl_mem_obj(i32, context, wflags, &out);
    defer _ = c.clReleaseMemObject(out_mem_obj);

    ret = c.clEnqueueWriteBuffer(command_queue, a_mem_obj, c.CL_TRUE, 0, @sizeOf(i32) * 128, &vec_a, 0, null, null);
    try check_error(ret);
    ret = c.clEnqueueWriteBuffer(command_queue, b_mem_obj, c.CL_TRUE, 0, @sizeOf(i32) * 128, &vec_b, 0, null, null);
    try check_error(ret);

    const program_source: [*c]const u8 = @embedFile("program.cl");
    const stupid_ptr = @constCast(&program_source);
    const program = c.clCreateProgramWithSource(context, 1, stupid_ptr, null, &ret);
    if (ret != 0) {
        std.log.err("Failed to create program: {d}", .{ret});
        return error.CreateProgramError;
    }

    defer _ = c.clReleaseProgram(program);

    ret = c.clBuildProgram(program, 1, &device_id, null, null, null);
    if (ret != 0) {
        std.log.err("Failed to build program: {d}", .{ret});
        return error.BuildProgramError;
    }

    const kernel = c.clCreateKernel(program, "vector_add", &ret);
    if (ret != 0) {
        std.log.err("Failed to create kernel: {d}", .{ret});
        return error.CreateKernelError;
    }

    defer _ = c.clReleaseKernel(kernel);

    ret = c.clSetKernelArg(kernel, 0, @sizeOf(c.cl_mem), a_mem_obj);
    try check_error(ret);
    ret = c.clSetKernelArg(kernel, 1, @sizeOf(c.cl_mem), b_mem_obj);
    try check_error(ret);
    ret = c.clSetKernelArg(kernel, 2, @sizeOf(c.cl_mem), out_mem_obj);
    try check_error(ret);

    const global_item_size: usize = 128; // Process the entire lists
    const local_item_size: usize = 4; // Divide work items into groups of 64

    ret = c.clEnqueueNDRangeKernel(command_queue, kernel, 1, null, &global_item_size, &local_item_size, 0, null, null);
    if (ret != 0) {
        std.log.err("Failed to enqueue kernel: {d}", .{ret});
        return error.EnqueueKernelError;
    }

    for (out) |val| {
        std.log.info("{}", .{val});
    }
}
