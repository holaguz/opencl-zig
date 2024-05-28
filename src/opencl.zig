const std = @import("std");
const c = @import("c.zig");
const Allocator = std.mem.Allocator;

pub const OpenCLContext = struct {
    const KernelContainer = std.StringArrayHashMap(c.cl_kernel);
    const BufferMeta = struct {
        buffer: c.cl_mem,
        size: usize,
        flags: c.cl_mem_flags,
        host_ptr: ?*anyopaque,
    };

    context: c.cl_context,
    command_queue: c.cl_command_queue,
    device_id: c.cl_device_id,
    program: c.cl_program,
    kernels: KernelContainer,
    buffers: std.ArrayList(BufferMeta),

    pub fn init(alloc: Allocator, device_id: c.cl_device_id, program_source: [*:0]const u8) !OpenCLContext {
        var ret: c.cl_int = undefined;
        const context = c.clCreateContext(0, 1, &device_id, cl_ctx_callback, null, &ret);
        try check_cl_error(ret, error.CreateContextError);
        errdefer _ = c.clReleaseContext(context);

        const command_queue = c.clCreateCommandQueueWithProperties(context, device_id, 0, &ret);
        try check_cl_error(ret, error.CreateCommandQueueError);
        errdefer _ = c.clReleaseCommandQueue(command_queue);

        const stupid_ptr: [*c]const u8 = @constCast(program_source);
        const program = c.clCreateProgramWithSource(context, 1, @constCast(&stupid_ptr), null, &ret);
        try check_cl_error(ret, error.CreateProgramError);

        const build_opts = "";
        ret = c.clBuildProgram(program, 1, &device_id, build_opts, null, null);
        try check_cl_error(ret, error.BuildProgramError);
        errdefer c.clReleaseProgram(program);

        const kernels = KernelContainer.init(alloc);
        const buffers = std.ArrayList(BufferMeta).init(alloc);

        return OpenCLContext{
            .context = context,
            .command_queue = command_queue,
            .device_id = device_id,
            .program = program,
            .kernels = kernels,
            .buffers = buffers,
        };
    }

    pub fn deinit(self: *OpenCLContext) void {
        _ = c.clReleaseContext(self.context);
        _ = c.clReleaseCommandQueue(self.command_queue);

        for (self.kernels.values()) |kernel| {
            _ = c.clReleaseKernel(kernel);
        }
        self.kernels.deinit();

        for (self.buffers.items) |buffer| {
            _ = c.clReleaseMemObject(buffer.buffer);
        }

        _ = c.clReleaseProgram(self.program);
    }

    pub fn get_kernel(self: *OpenCLContext, kernel_name: []const u8) ?c.cl_kernel {
        return self.kernels.get(kernel_name);
    }

    pub fn create_kernel(self: *OpenCLContext, kernel_name: []const u8) !c.cl_kernel {
        if (self.kernels.get(kernel_name)) |_| {
            return error.KernelExists;
        }

        var ret: c.cl_int = undefined;
        const kernel = c.clCreateKernel(self.program, kernel_name.ptr, &ret);
        try check_cl_error(ret, error.CreateKernelError);
        try self.kernels.put(kernel_name, kernel);
        return kernel;
    }

    pub fn create_buffer(self: *OpenCLContext, size: usize, flags: c.cl_mem_flags, host_ptr: ?*anyopaque) !c.cl_mem {
        var ret: c.cl_int = undefined;
        const buffer_obj = c.clCreateBuffer(self.context, flags, size, host_ptr, &ret);
        try check_cl_error(ret, error.CreateBufferError);
        try self.buffers.append(.{
            .buffer = buffer_obj,
            .size = size,
            .flags = flags,
            .host_ptr = host_ptr,
        });
        return buffer_obj;
    }

    pub fn write_buffer(self: *OpenCLContext, comptime T: type, buffer: c.cl_mem, data: []const T) !void {
        var ret: c.cl_int = undefined;
        const size = data.len * @sizeOf(T);
        ret = c.clEnqueueWriteBuffer(self.command_queue, buffer, c.CL_TRUE, 0, size, data.ptr, 0, null, null);
        try check_cl_error(ret, error.WriteBufferError);
    }

    pub fn read_buffer(self: *OpenCLContext, comptime T: type, buffer: c.cl_mem, data: []T) !void {
        var ret: c.cl_int = undefined;
        const size = data.len * @sizeOf(T);
        ret = c.clEnqueueReadBuffer(self.command_queue, buffer, c.CL_TRUE, 0, size, data.ptr, 0, null, null);
        try check_cl_error(ret, error.ReadBufferError);
    }

    pub fn set_kernel_arg(self: *OpenCLContext, kernel: c.cl_kernel, arg_index: c.cl_uint, arg_value: anytype) !void {
        _ = self;
        comptime {
            const type_info = @typeInfo(@TypeOf(arg_value));
            if (@as(std.builtin.Type, type_info) != std.builtin.TypeId.Pointer) {
                @compileError("Argument must be a pointer");
            }
        }

        var ret: c.cl_int = undefined;
        const arg_size = @sizeOf(@typeInfo(@TypeOf(arg_value)).Pointer.child);
        ret = c.clSetKernelArg(kernel, arg_index, arg_size, @ptrCast(arg_value));
        try check_cl_error(ret, error.SetKernelArgError);
    }

    pub fn run_kernel(self: *OpenCLContext, block: bool, kernel: c.cl_kernel, work_dim: u32, global_work_offset: usize, global_work_size: usize, local_item_size: usize) !void {
        var ret: c.cl_int = undefined;
        ret = c.clEnqueueNDRangeKernel(self.command_queue, kernel, work_dim, &global_work_offset, &global_work_size, &local_item_size, 0, null, null);
        try check_cl_error(ret, error.EnqueueKernel);

        if (block) {
            ret = c.clFinish(self.command_queue);
            try check_cl_error(ret, error.KernelFinish);
        }
    }
};

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

pub fn cl_ctx_callback(error_info: [*c]const u8, private_info: ?*const anyopaque, cb: usize, user_data: ?*anyopaque) callconv(.C) void {
    _ = user_data; // autofix
    _ = cb; // autofix
    _ = private_info; // autofix

    std.log.info("cl_ctx_callback: {s}", .{error_info});
}

pub fn enumerate() c.cl_device_id {
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

    var device_name: [64]u8 = undefined;
    var device_vendor: [64]u8 = undefined;
    _ = c.clGetDeviceInfo(device_id, c.CL_DEVICE_NAME, 64, &device_name, null);
    _ = c.clGetDeviceInfo(device_id, c.CL_DEVICE_VENDOR, 64, &device_vendor, null);

    std.log.debug("Selecting first device:", .{});
    std.log.debug(" - Vendor: {s}", .{device_vendor});
    std.log.debug(" - Name:   {s}", .{device_name});

    return device_id;
}
