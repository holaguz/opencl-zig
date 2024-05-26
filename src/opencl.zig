const std = @import("std");
const c = @import("c.zig");
const Allocator = std.mem.Allocator;

pub const OpenCLContext = struct {
    const KernelContainer = std.StringArrayHashMap(c.cl_kernel);

    context: c.cl_context,
    command_queue: c.cl_command_queue,
    device_id: c.cl_device_id,
    program: c.cl_program,
    kernels: KernelContainer,

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

        return OpenCLContext{
            .context = context,
            .command_queue = command_queue,
            .device_id = device_id,
            .program = program,
            .kernels = kernels,
        };
    }

    pub fn deinit(self: *OpenCLContext) void {
        c.clReleaseContext(self.context);
        c.clReleaseCommandQueue(self.command_queue);

        for (self.kernels.items()) |kernel| {
            c.clReleaseKernel(kernel);
        }
        self.kernels.deinit();

        c.clReleaseProgram(self.program);
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

    // pub fn load_kernel_args();
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
