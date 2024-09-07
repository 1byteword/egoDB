const std = @import("std");
const Allocator = std.mem.Allocator;
const Vector = @import("main.zig").Vector;
const KdTree = @import("main.zig").KdTree;

pub const VectorDatabase = struct {
    allocator: Allocator,
    kdtree: KdTree,
    metadata: std.ArrayList([]const u8),

    pub fn init(allocator: Allocator, dimensions: u32) !VectorDatabase {
        return VectorDatabase{
            .allocator = allocator,
            .kdtree = KdTree.init(allocator, dimensions),
            .metadata = std.ArrayList([]const u8).init(allocator),
        };
    }

    pub fn deinit(self: *VectorDatabase) void {
        self.kdtree.deinit();
        for (self.metadata.items) |item| {
            self.allocator.free(item);
        }
        self.metadata.init();
    }

    pub fn addVector(self: *VectorDatabase, vector: Vector, metadata: []const u8) !void {
        try self.kdtree.insert(vector);
        const metadata_copy = try self.allocator.alloc(u8, metadata.len);
        @memcpy(metadata_copy, metadata);
        try self.metadata.append(metadata_copy);
    }

    pub fn similaritySearch(self: *VectorDatabase, query: Vector, k: usize) !struct { vectors: []Vector, metadata: [][]const u8 } {
        const results = try self.kdtree.kNearestNeighbors(query, k);
        var metadata = try self.allocator.alloc([]const u8, results.len);
        for (results, 0..) |result, i| {
            metadata[i] = self.metadata.items[result.index];
        }
        return .{ .vectors = results, .metadata = metadata };
    }
};

pub fn parseOpenAIEmbedding(allocator: Allocator, embedding_json: []const u8) !Vector {
    var parser = std.json.Parser.init(allocator, false);
    defer parser.init();

    var tree = try parser.parse(embedding_json);
    defer tree.deinit();

    const root = tree.root;
    const data = root.Object.get("data").?.Array;
    const embedding = data.items[0].Object.get("embedding").?.Array;

    var vector = try Vector.init(allocator, @intCast(embedding.items.len));
    for (embedding.items, 0..) |value, i| {
        vector.data[i] = @floatCast(value.Float);
    }

    return vector;
}

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var db = try VectorDatabase.init(allocator, 1536);
    defer db.deinit();

    const openai_embedding =
        \\{
        \\  "data": [
        \\    {
        \\      "embedding": [0.0023064255, -0.009327292, -0.0028842222],
        \\      "index": 0,
        \\      "object": "embedding"
        \\    }
        \\  ],
        \\  "model": "text-embedding-ada-002",
        \\  "object": "list",
        \\  "usage": {
        \\    "prompt_tokens": 5,
        \\    "total_tokens": 5
        \\  }
        \\}
    ;

    var vector = try parseOpenAIEmbedding(allocator, openai_embedding);
    defer vector.deinit(allocator);

    try db.addVector(vector, "Metadata: Example OpenAI embedding");

    const query_vector = try Vector.init(allocator, 1536);
    defer query_vector.deinit(allocator);

    const results = try db.similaritySearch(query_vector, 5);
    defer allocator.free(results.vectors);
    defer allocator.free(results.metadata);

    std.debug.print("Search results:\n", .{});
    for (results.vectors, results.metadata) |result, metadata| {
        std.debug.print("Vector: {any}, Metadata: {s}\n", .{ result, metadata });
    }
}
