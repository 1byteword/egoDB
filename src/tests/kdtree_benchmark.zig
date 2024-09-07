const std = @import("std");
const time = std.time;
const Vector = @import("../main.zig").Vector;
const KdTree = @import("../main.zig").KdTree;

const SimpleRng = struct {
    state: u64,

    fn init(seed: u64) SimpleRng {
        return SimpleRng{ .state = seed };
    }

    fn next(self: *SimpleRng) u64 {
        self.state = self.state *% 6364136223846793005 +% 1;
        return self.state;
    }

    fn float(self: *SimpleRng) f32 {
        return @as(f32, @floatFromInt(self.next() & 0x7FFFFFFF)) / @as(f32, @floatFromInt(0x7FFFFFFF));
    }
};

pub fn run() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Parameters for stress test
    const dimensions: u32 = 1536; // OpenAI embedding size
    const num_points: usize = 10_000; // Reduced due to higher dimensionality
    const num_searches: usize = 100; // Reduced for quicker overall runtime

    std.debug.print("Starting KdTree stress test and benchmark\n", .{});
    std.debug.print("Dimensions: {}, Points: {}, Searches: {}\n", .{ dimensions, num_points, num_searches });

    var rng = SimpleRng.init(@intCast(std.time.milliTimestamp()));

    // Generate random points
    const points = try allocator.alloc(*Vector, num_points);
    defer {
        for (points) |point| {
            point.deinit(allocator);
            allocator.destroy(point);
        }
        allocator.free(points);
    }

    for (points) |*point_ptr| {
        point_ptr.* = try allocator.create(Vector);
        point_ptr.*.* = try generateRandomVector(allocator, dimensions, &rng);
    }

    // Create and populate KdTree
    var kdtree = KdTree.init(allocator, dimensions);
    defer kdtree.deinit();

    const insert_start = time.milliTimestamp();
    for (points) |point| {
        try kdtree.insert(point.*);
    }
    const insert_end = time.milliTimestamp();
    const insert_duration = @as(f64, @floatFromInt(insert_end - insert_start)) / 1000.0;

    std.debug.print("Insertion of {} points took {d:.6} seconds\n", .{ num_points, insert_duration });

    // Perform nearest neighbor searches
    const search_start = time.milliTimestamp();
    for (0..num_searches) |_| {
        var target = try generateRandomVector(allocator, dimensions, &rng);
        defer target.deinit(allocator);

        _ = kdtree.nearestNeighbor(target);
    }
    const search_end = time.milliTimestamp();
    const search_duration = @as(f64, @floatFromInt(search_end - search_start)) / 1000.0;

    std.debug.print("Performing {} nearest neighbor searches took {d:.6} seconds\n", .{ num_searches, search_duration });

    const avg_search_time = search_duration / @as(f64, @floatFromInt(num_searches));
    std.debug.print("Average time per search: {d:.6} milliseconds\n", .{avg_search_time * 1000.0});
}

fn generateRandomVector(allocator: std.mem.Allocator, dimensions: u32, rng: *SimpleRng) !Vector {
    var vec = try Vector.init(allocator, dimensions);
    for (0..dimensions) |i| {
        vec.data[i] = rng.float() * 100.0; // Random float between 0 and 100
    }
    return vec;
}
