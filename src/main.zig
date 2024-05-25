const std = @import("std");
const c = @import("c.zig");

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

pub fn main() !void {
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

    const cl_ctx = c.clCreateContext(0, 1, &device_id, cl_ctx_callback, null, &ret);
    if (ret != 0) {
        std.log.err("Failed to create cl_context: {d}", .{ret});
        return error.CreateContextError;
    }

    const command_queue = c.clCreateCommandQueueWithProperties(cl_ctx, device_id, 696969, &ret);
    _ = command_queue;
    if (ret != 0) {
        std.log.err("Failed to create command queue: {d}", .{ret});
        return error.CreateCmdQueueError;
    }
}
