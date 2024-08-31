//! By convention, main.zig is where your main function lives in the case that
//! you are building an executable. If you are making a library, the convention
//! is to delete this file and start with root.zig instead.
const std = @import("std");
const Allocator = std.mem.Allocator;

fn abs(x: f32) f32 {
    return if (x < 0) -x else x;
}

pub const KdTree = struct {
    const Node = struct {
        point: Vector,
        left: ?*Node,
        right: ?*Node,
        split_dim: u32,
    };

    root: ?*Node,
    allocator: Allocator,
    dimensions: u32,

    pub fn init(allocator: Allocator, dimensions: u32) KdTree {
        std.debug.print("Initializing KdTree with dimensions: {}\n", .{dimensions});
        return .{
            .root = null,
            .allocator = allocator,
            .dimensions = dimensions,
        };
    }

    pub fn insert(self: *KdTree, point: Vector) !void {
        std.debug.print("Inserting point: ", .{});
        printVector(point);
        if (point.dimensions != self.dimensions) return error.InvalidDimension;
        self.root = try self.insertRec(self.root, point, 0);
    }

    fn insertRec(self: *KdTree, node: ?*Node, point: Vector, depth: u32) !?*Node {
        if (node == null) {
            const new_node = try self.allocator.create(Node);
            errdefer self.allocator.destroy(new_node);
            new_node.* = .{
                .point = try point.clone(self.allocator),
                .left = null,
                .right = null,
                .split_dim = depth % self.dimensions,
            };
            std.debug.print("Created new node at depth {}\n", .{depth});
            return new_node;
        }

        const curr_node = node.?;
        const split_dim = depth % self.dimensions;

        if (point.data[split_dim] < curr_node.point.data[split_dim]) {
            curr_node.left = try self.insertRec(curr_node.left, point, depth + 1);
        } else {
            curr_node.right = try self.insertRec(curr_node.right, point, depth + 1);
        }

        return curr_node;
    }

    pub fn nearestNeighbor(self: *KdTree, target: Vector) ?Vector {
        std.debug.print("Searching for nearest neighbor to: ", .{});
        printVector(target);
        if (self.root == null) return null;
        var best = self.root.?.point;
        var best_dist = target.euclideanDistance(best);
        self.nearestNeighborRec(self.root.?, target, 0, &best, &best_dist);
        return best;
    }

    fn nearestNeighborRec(self: *KdTree, node: *Node, target: Vector, depth: u32, best: *Vector, best_dist: *f32) void {
        const split_dim = depth % self.dimensions;
        const dist = target.euclideanDistance(node.point);

        if (dist < best_dist.*) {
            best.* = node.point;
            best_dist.* = dist;
        }

        const next_branch = if (target.data[split_dim] < node.point.data[split_dim]) node.left else node.right;
        const other_branch = if (target.data[split_dim] < node.point.data[split_dim]) node.right else node.left;

        if (next_branch) |branch| {
            self.nearestNeighborRec(branch, target, depth + 1, best, best_dist);
        }

        const dist_split = abs(target.data[split_dim] - node.point.data[split_dim]);
        if (dist_split < best_dist.*) {
            if (other_branch) |branch| {
                self.nearestNeighborRec(branch, target, depth + 1, best, best_dist);
            }
        }
    }

    pub fn deinit(self: *KdTree) void {
        if (self.root) |root| {
            self.deinitRec(root);
        }
    }

    fn deinitRec(self: *KdTree, node: *Node) void {
        if (node.left) |left| self.deinitRec(left);
        if (node.right) |right| self.deinitRec(right);
        node.point.deinit(self.allocator);
        self.allocator.destroy(node);
    }
};

pub const Vector = struct {
    dimensions: u32,
    data: []f32,

    // init Vector in row-wise
    // (i.e. [1 2 3
    //        4 5 6
    //        7 8 9]
    //  would be read in left to right 1 2 3, then wrap around to 4 5 6, and so on.)
    pub fn init(allocator: std.mem.Allocator, dimensions: u32) !Vector {
        const data = try allocator.alloc(f32, dimensions);
        return Vector{
            .dimensions = dimensions,
            .data = data,
        };
    }

    // free up space from heap
    pub fn deinit(self: *Vector, allocator: std.mem.Allocator) void {
        allocator.free(self.data);
    }

    pub fn clone(self: Vector, allocator: Allocator) !Vector {
        const new_vector = try Vector.init(allocator, self.dimensions);
        @memcpy(new_vector.data, self.data);
        return new_vector;
    }

    //implement all vector operations
    pub fn add(self: Vector, other: Vector, allocator: std.mem.Allocator) !Vector {
        std.debug.assert(self.dimensions == other.dimensions);
        var result = try Vector.init(allocator, self.dimensions);
        for (self.data, other.data, 0..) |a, b, i| {
            result.data[i] = a + b;
        }
        return result;
    }

    pub fn subtract(self: Vector, other: Vector, allocator: std.mem.Allocator) !Vector {
        std.debug.assert(self.dimensions == other.dimensions);
        var result = try Vector.init(allocator, self.dimensions);
        for (self.data, other.data, 0..) |a, b, i| {
            result.data[i] = a + b;
        }
        return result;
    }

    pub fn scale(self: Vector, scalar: f32, allocator: std.mem.Allocator) !Vector {
        var result = try Vector.init(allocator, self.dimensions);
        for (self.data, 0..) |a, i| {
            result.data[i] = a * scalar;
        }
        return result;
    }

    // dot product
    pub fn dot(self: Vector, other: Vector) f32 {
        std.debug.assert(self.dimensions == other.dimensions);
        var result: f32 = 0;
        for (self.data, other.data) |a, b| {
            result += (a * b);
        }
        return result;
    }

    // return scalar magnitude
    pub fn magnitude(self: Vector) f32 {
        var result: f32 = 0;
        for (self.data) |a| {
            result += (a * a);
        }
        return @sqrt(result);
    }

    // return a new vector, remember to de-allocate
    pub fn normalize(self: Vector, allocator: std.mem.Allocator) !Vector {
        var result = try Vector.init(allocator, self.dimensions);
        const mag = self.magnitude();
        std.debug.assert(mag != 0);
        for (self.data, 0..) |a, i| {
            result.data[i] = a / mag;
        }
        return result;
    }

    pub fn cosineSimilarity(self: Vector, other: Vector, allocator: std.mem.Allocator) f32 {
        var normalizedSelf = try self.normalize(allocator);
        defer normalizedSelf.deinit(allocator);

        var normalizedOther = try other.normalize(allocator);
        defer normalizedOther.deinit(allocator);

        return self.dot(other);
    }

    pub fn euclideanDistance(self: Vector, other: Vector) f32 {
        var distance: f32 = 0;
        for (self.data, other.data) |a, b| {
            distance += (a - b) * (a - b);
        }
        return @sqrt(distance);
    }
};

pub fn main() !void {
    std.debug.print("Spinning up our EgoDB...\n", .{});
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var kdtree = KdTree.init(allocator, 3);
    defer kdtree.deinit();

    const points = [_][]const f32{
        &[_]f32{ 1, 2, 3 },
        &[_]f32{ 4, 5, 6 },
        &[_]f32{ 7, 8, 9 },
        &[_]f32{ 2, 3, 4 },
        &[_]f32{ 5, 6, 7 },
    };

    for (points) |point_data| {
        var point = try Vector.init(allocator, 3);
        defer point.deinit(allocator);
        @memcpy(point.data, point_data);
        try kdtree.insert(point);
    }

    // Find nearest neighbor
    var target = try Vector.init(allocator, 3);
    defer target.deinit(allocator);
    target.data[0] = 3;
    target.data[1] = 4;
    target.data[2] = 5;

    std.debug.print("Finding nearest neighbor for: ", .{});
    printVector(target);

    if (kdtree.nearestNeighbor(target)) |nearest| {
        std.debug.print("Nearest neighbor: ", .{});
        printVector(nearest);
    } else {
        std.debug.print("No nearest neighbor found.\n", .{});
    }
}

fn printVector(v: Vector) void {
    for (v.data) |value| {
        std.debug.print("{d:.2} ", .{value});
    }
    std.debug.print("\n", .{});
}
