/**
 * @file cudaQuadTree.cu
 * @brief Implementation of an asynchronous spatial query system using CUDA and quadtrees
 * 
 * This file implements a parallel spatial query system that can efficiently find
 * intersecting rectangles in 2D space. It uses a quadtree data structure for
 * spatial partitioning and CUDA for parallel processing.
 */

#pragma nv_disjoint_regions 
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <iostream>
#include <vector>
#include <random>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
//#define printf(fmt, ...) (0)

// Configuration constants
const int MAX_RESULTS = 1000;  // Maximum number of intersection results to store
const int NUM_RECTS = 1000;    // Total number of rectangles to process

/**
 * @struct Rect
 * @brief Represents a 2D rectangle with tracking information
 * 
 * Stores the coordinates of a rectangle (min/max points) and maintains
 * its original index in the input array for result tracking.
 */
struct Rect {
    int min_x, min_y, max_x, max_y;
    int original_index;  // Preserves the original position in input array
    
    __host__ __device__ Rect() : original_index(-1) {}
    
    __host__ __device__ Rect(int _min_x, int _min_y, int _max_x, int _max_y)
        : min_x(_min_x), min_y(_min_y), max_x(_max_x), max_y(_max_y), original_index(-1) {}
    
    /**
     * @brief Calculates the center point of the rectangle
     * @return float2 containing the x,y coordinates of the center
     */
    __host__ __device__ float2 get_center() const {
        return make_float2((min_x + max_x) * 0.5f,
                          (min_y + max_y) * 0.5f);
    }
    
    /**
     * @brief Checks if this rectangle intersects with another
     * @param other The rectangle to check intersection with
     * @return true if rectangles intersect, false otherwise
     */
    __host__ __device__ bool intersects(const Rect& other) const {
        return !(max_x <= other.min_x || 
                other.max_x <= min_x ||
                max_y <= other.min_y || 
                other.max_y <= min_y);
    }
};

/**
 * @struct Bounding_box
 * @brief Represents an axis-aligned bounding box for quadtree nodes
 * 
 * Used to efficiently perform spatial queries and determine which nodes
 * need to be traversed during searches.
 */
struct Bounding_box {
    int m_min_x, m_min_y, m_max_x, m_max_y;

public:
    __host__ __device__ Bounding_box() {}
    
    __host__ __device__ void set(int min_x, int min_y, int max_x, int max_y) {
        m_min_x = min_x;
        m_min_y = min_y;
        m_max_x = max_x;
        m_max_y = max_y;
    }
    
    /**
     * @brief Tests intersection between bounding box and rectangle
     * @param rect Rectangle to test against
     * @return true if there is an intersection, false otherwise
     */
    __host__ __device__ bool intersects(const Rect& rect) const {
        return !(m_max_x < rect.min_x || rect.max_x < m_min_x ||
                m_max_y < rect.min_y || rect.max_y < m_min_y);
    }
    
    __host__ __device__ float2 get_center() const {
        return make_float2((m_min_x + m_max_x) * 0.5f,
                          (m_min_y + m_max_y) * 0.5f);
    }
};

/**
 * @class Quadtree_node
 * @brief Represents a node in the quadtree spatial index
 * 
 * Each node contains a bounding box and either subdivides into four children
 * or contains a list of rectangles. The quadtree automatically subdivides
 * when a node contains too many rectangles.
 */
class Quadtree_node {
    Bounding_box m_bbox;      // Spatial bounds of this node
    int m_begin;             // Start index of rectangles in this node
    int m_end;              // End index of rectangles in this node
    bool m_is_leaf;         // True if this is a leaf node (contains rectangles)
    int m_child_offset;     // Index offset to first child node (if not leaf)

public:
    __host__ __device__ Quadtree_node()
        : m_begin(0), m_end(0), m_is_leaf(true), m_child_offset(-1) {}
        
    // Accessor methods
    __host__ __device__ void set_range(int begin, int end) {
        m_begin = begin;
        m_end = end;
    }
    
    __host__ __device__ int points_begin() const { return m_begin; }
    __host__ __device__ int points_end() const { return m_end; }
    __host__ __device__ bool is_leaf() const { return m_is_leaf; }
    __host__ __device__ void set_leaf(bool is_leaf) { m_is_leaf = is_leaf; }
    __host__ __device__ const Bounding_box& bounding_box() const { return m_bbox; }
    __host__ __device__ Bounding_box& bounding_box() { return m_bbox; }
    __host__ __device__ void set_child_offset(int offset) { m_child_offset = offset; }
    __host__ __device__ int child_offset() const { return m_child_offset; }
};

/**
 * @brief CUDA device function to recursively search the quadtree
 * 
 * @param nodes Array of quadtree nodes
 * @param rects Array of rectangles being searched
 * @param node_idx Current node index
 * @param query_rect Rectangle to search for intersections with
 * @param result_indices Output array for storing intersecting rectangle indices
 * @param result_count Atomic counter for number of results found
 * @param max_results Maximum number of results to store
 */
__device__ void search_node(
    const Quadtree_node* nodes,
    const Rect* rects,
    int node_idx,
    const Rect& query_rect,
    int* result_indices,
    int* result_count,
    int max_results
) {
    const Quadtree_node& node = nodes[node_idx];
    
    // Early exit if node's bounding box doesn't intersect query
    if (!node.bounding_box().intersects(query_rect)) {
        return;
    }
    
    if (node.is_leaf()) {
        // For leaf nodes, check all contained rectangles
        for (int i = node.points_begin(); i < node.points_end(); i++) {
            const Rect& rect = rects[i];
            
            if (rect.intersects(query_rect)) {
                // Atomically add result to output array
                int idx = atomicAdd(result_count, 1);
                if (idx < max_results) {
                    result_indices[idx] = rect.original_index;
                }
            }
        }
    } else {
        // For internal nodes, recursively search all children
        for (int i = 0; i < 4; i++) {
            int child_idx = node.child_offset() + i;
            if (child_idx < NUM_RECTS * 4) {  // Bounds check
                search_node(nodes, rects, child_idx,
                           query_rect, result_indices, result_count, max_results);
            }
        }
    }
}

/**
 * @brief CUDA kernel for building the quadtree structure
 * 
 * This kernel recursively subdivides nodes that contain too many rectangles.
 * It uses CUDA Dynamic Parallelism (CDP) for recursive subdivision.
 * 
 * @param nodes Array of quadtree nodes
 * @param rects Array of rectangles to organize
 * @param node_idx Current node being processed
 * @param depth Current depth in the tree
 * @param max_depth Maximum allowed tree depth
 * @param min_rects_per_node Minimum rectangles before subdivision
 * @param max_nodes Maximum number of nodes allowed in the tree
 */
__global__ void build_quadtree_kernel(
    Quadtree_node* nodes,
    Rect* rects,
    int node_idx,
    int depth,
    int max_depth,
    int min_rects_per_node,
    int max_nodes
) {
    if (node_idx >= max_nodes) return;
    
    // Initialize root node's bounding box (only done once at depth 0)
    if (depth == 0 && threadIdx.x == 0 && blockIdx.x == 0) {
        Quadtree_node& root = nodes[0];
        
        // Find bounding box that contains all rectangles
        int min_x = INT_MAX, min_y = INT_MAX;
        int max_x = INT_MIN, max_y = INT_MIN;
        
        for (int i = root.points_begin(); i < root.points_end(); i++) {
            min_x = min(min_x, rects[i].min_x);
            min_y = min(min_y, rects[i].min_y);
            max_x = max(max_x, rects[i].max_x);
            max_y = max(max_y, rects[i].max_y);
        }
        
        root.bounding_box().set(min_x, min_y, max_x, max_y);
    }

    Quadtree_node& node = nodes[node_idx];
    int num_rects = node.points_end() - node.points_begin();
    
    // Stop subdivision if we've reached max depth or have few enough rectangles
    if (depth >= max_depth || num_rects <= min_rects_per_node) {
        if (threadIdx.x == 0 && blockIdx.x == 0) {
            node.set_leaf(true);
            printf("  -> Leaf node (depth=%d, rects=%d)\n", depth, num_rects);
        }
        return;
    }
    
    // Only one thread performs the subdivision
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        const auto& bbox = node.bounding_box();
        
        // Calculate center point for subdivision
        int center_x = (bbox.m_min_x + bbox.m_max_x) / 2;
        int center_y = (bbox.m_min_y + bbox.m_max_y) / 2;
    
        // Adjust center points to prevent degenerate cases
        if (center_x == bbox.m_min_x) center_x++;
        if (center_y == bbox.m_min_y) center_y++;
        if (center_x == bbox.m_max_x) center_x--;
        if (center_y == bbox.m_max_y) center_y--;
        
        // Count rectangles that will go in each quadrant
        int counts[4] = {0, 0, 0, 0};
        
        printf("  Rectangles in this node:\n");
        for (int i = node.points_begin(); i < node.points_end(); i++) {
            const Rect& rect = rects[i];
            int rect_center_x = (rect.min_x + rect.max_x) / 2;
            int rect_center_y = (rect.min_y + rect.max_y) / 2;
        
            printf("    Rect %d: (%d,%d)-(%d,%d), center=(%d,%d)\n",
                   i, rect.min_x, rect.min_y, rect.max_x, rect.max_y,
                   rect_center_x, rect_center_y);
        
            // Determine which quadrant the rectangle belongs in
            int quad = (rect_center_x >= center_x ? 1 : 0) +
                      (rect_center_y >= center_y ? 2 : 0);
            counts[quad]++;
        }
    
        printf("  Quadrant counts: [%d, %d, %d, %d]\n",
            counts[0], counts[1], counts[2], counts[3]);
            
        // Set up node as internal node
        node.set_leaf(false);
        node.set_child_offset(node_idx * 4 + 1);
        
        // Calculate starting positions for each quadrant
        int positions[4];
        positions[0] = node.points_begin();
        for (int i = 1; i < 4; i++) {
            positions[i] = positions[i - 1] + counts[i - 1];
        }

        // Debug output for subdivision process
        printf("  Split positions: [%d, %d, %d, %d]\n",
            positions[0], positions[1], positions[2], positions[3]);
        printf("  Reordering rectangles...\n");
        
        // Partition rectangles into quadrants
        for (int quad = 0; quad < 4; quad++) {
            printf("\n  Processing quadrant %d (target range: [%d,%d))...\n", 
                   quad, positions[quad], positions[quad] + counts[quad]);
    
            int pos = positions[quad];
            // Move rectangles to their correct quadrants
            for (int i = pos; i < node.points_end(); i++) {
                const Rect& rect = rects[i];
                int rect_center_x = (rect.min_x + rect.max_x) / 2;
                int rect_center_y = (rect.min_y + rect.max_y) / 2;
                int curr_quad = (rect_center_x >= center_x ? 1 : 0) +
                              (rect_center_y >= center_y ? 2 : 0);
        
                if (curr_quad == quad && pos < positions[quad] + counts[quad]) {
                    if (i != pos) {
                        // Swap rectangles to move them to correct position
                        Rect temp = rects[pos];
                        rects[pos] = rects[i];
                        rects[i] = temp;
                    }
                    pos++;
                }
            }
        }

        // Initialize child nodes
        for (int i = 0; i < 4; i++) {
            Quadtree_node& child = nodes[node.child_offset() + i];
            child.set_range(positions[i], i < 3 ? positions[i + 1] : node.points_end());
            
            // Calculate child node's bounding box
            int child_min_x = INT_MAX, child_min_y = INT_MAX;
            int child_max_x = INT_MIN, child_max_y = INT_MIN;
        
            // Find bounds of rectangles in this child
            for (int j = child.points_begin(); j < child.points_end(); j++) {
                const Rect& rect = rects[j];
                child_min_x = min(child_min_x, rect.min_x);
                child_min_y = min(child_min_y, rect.min_y);
                child_max_x = max(child_max_x, rect.max_x);
                child_max_y = max(child_max_y, rect.max_y);
            }
        
            // Handle empty nodes
            if (child.points_begin() == child.points_end()) {
                child_min_x = center_x;
                child_min_y = center_y;
                child_max_x = center_x + 1;
                child_max_y = center_y + 1;
            }
        
            // Ensure valid bounding box
            if (child_min_x >= child_max_x) child_max_x = child_min_x + 1;
            if (child_min_y >= child_max_y) child_max_y = child_min_y + 1;
        
            child.bounding_box().set(child_min_x, child_min_y, 
                                   child_max_x, child_max_y);
        
            // Recursively build child nodes using CUDA Dynamic Parallelism
            if (child.points_end() > child.points_begin()) {
                build_quadtree_kernel<<<1, 1>>>(
                    nodes,
                    rects,
                    node.child_offset() + i,
                    depth + 1,
                    max_depth,
                    min_rects_per_node,
                    max_nodes
                );
            }
        }
    }
}

/**
 * @brief CUDA kernel for searching rectangles in the quadtree
 * 
 * @param nodes Array of quadtree nodes
 * @param rects Array of rectangles
 * @param query_rect The rectangle to search for intersections with
 * @param result_indices Output array for storing intersecting rectangle indices
 * @param result_count Counter for number of results found
 * @param max_results Maximum number of results to store
 */
__global__ void search_rect_kernel(
    const Quadtree_node* nodes,
    const Rect* rects,
    const Rect query_rect,
    int* result_indices,
    int* result_count,
    int max_results
) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *result_count = 0;
        search_node(nodes, rects, 0, query_rect,
                   result_indices, result_count, max_results);
    }
}

/**
 * @brief Helper function to check if two rectangles overlap
 */
bool rectangles_overlap(const Rect& r1, const Rect& r2) {
    return !(r1.max_x <= r2.min_x || 
            r2.max_x <= r1.min_x ||
            r1.max_y <= r2.min_y || 
            r2.max_y <= r1.min_y);
}

/**
 * @brief CPU function to verify search results
 * 
 * Checks all results for validity and identifies any missed intersections
 */
void verify_results(const std::vector<Rect>& all_rects, 
                   const std::vector<int>& result_indices,
                   int result_count,
                   const Rect& query_rect) {
    std::cout << "\n=== Verification Results ===\n";
    std::cout << "Query rectangle: (" 
              << query_rect.min_x << "," << query_rect.min_y << ") - ("
              << query_rect.max_x << "," << query_rect.max_y << ")\n\n";

    // Verify each reported intersection
    int valid_intersections = 0;
    for (int i = 0; i < result_count; i++) {
        int idx = result_indices[i];
        if (idx >= 0 && idx < all_rects.size()) {
            const Rect& rect = all_rects[idx];
            bool does_overlap = rectangles_overlap(rect, query_rect);
            
            std::cout << "Result " << i << " (Rectangle " << idx << "): ("
                     << rect.min_x << "," << rect.min_y << ") - ("
                     << rect.max_x << "," << rect.max_y << ") "
                     << (does_overlap ? "VALID" : "INVALID") << "\n";
                     
            if (does_overlap) valid_intersections++;
        }
    }
    
    // Check for missed intersections
    std::cout << "\nChecking for missed intersections...\n";
    int missed_intersections = 0;
    for (int i = 0; i < all_rects.size(); i++) {
        bool found_in_results = false;
        for (int j = 0; j < result_count; j++) {
            if (result_indices[j] == i) {
                found_in_results = true;
                break;
            }
        }
        
        if (!found_in_results && rectangles_overlap(all_rects[i], query_rect)) {
            std::cout << "Missed Rectangle " << i << ": ("
                     << all_rects[i].min_x << "," << all_rects[i].min_y << ") - ("
                     << all_rects[i].max_x << "," << all_rects[i].max_y << ")\n";
            missed_intersections++;
        }
    }
    
    // Output summary statistics
    std::cout << "\n=== Summary ===\n";
    std::cout << "Total results: " << result_count << "\n";
    std::cout << "Valid intersections: " << valid_intersections << "\n";
    std::cout << "Invalid results: " << (result_count - valid_intersections) << "\n";
    std::cout << "Missed intersections: " << missed_intersections << "\n";
}

/**
 * @brief CPU implementation of quadtree search with cycle detection
 */
void search_node_cpu(const Quadtree_node* nodes, 
                    const Rect* rects,
                    int node_idx,
                    const Rect& query_rect,
                    std::vector<int>& result_indices,
                    int max_node_idx,
                    std::vector<bool>& visited) {
    // Check for cycles in the tree traversal
    if (visited[node_idx]) {
        std::cout << "Cycle detected at node " << node_idx << "\n";
        return;
    }
    visited[node_idx] = true;

    // Validate node index
    if (node_idx < 0 || node_idx >= max_node_idx) {
        std::cout << "Warning: Invalid node index " << node_idx 
                 << " (max: " << max_node_idx - 1 << ")\n";
        return;
    }

    const Quadtree_node& node = nodes[node_idx];

    // Validate child node indices
    if (!node.is_leaf() && 
        (node.child_offset() < 0 || 
         node.child_offset() + 4 > max_node_idx)) {
        std::cout << "Warning: Invalid child offset " << node.child_offset() 
                 << " at node " << node_idx << "\n";
        return;
    }

    // Create rectangle from node's bounding box for intersection test
    Rect node_bbox(node.bounding_box().m_min_x,
                  node.bounding_box().m_min_y,
                  node.bounding_box().m_max_x,
                  node.bounding_box().m_max_y); 

    if (!rectangles_overlap(node_bbox, query_rect)) {
        return;
    }
    
    if (node.is_leaf()) {
        // Process rectangles in leaf node
        for (int i = node.points_begin(); i < node.points_end(); i++) {
            if (rectangles_overlap(rects[i], query_rect)) {
                result_indices.push_back(rects[i].original_index);
            }
        }
    } else {
        // Debug output for tree traversal
        std::cout << "Node " << node_idx << " (bbox: " << node_bbox.min_x 
                 << "," << node_bbox.min_y << "-" << node_bbox.max_x 
                 << "," << node_bbox.max_y << ") children: ";
        // Recursively process child nodes
        for (int i = 0; i < 4; i++) {
            int child_idx = node.child_offset() + i;
            std::cout << child_idx << " ";
            if (child_idx < max_node_idx) {
                search_node_cpu(nodes, rects, child_idx, query_rect, 
                              result_indices, max_node_idx, visited);
            }
        }
        std::cout << "\n";
    }
}

/**
 * @brief Wrapper function for CPU quadtree search
 */
std::vector<int> search_rect_cpu(const Quadtree_node* nodes,
                                const Rect* rects,
                                const Rect& query_rect,
                                int max_node_idx) {
    std::vector<int> result_indices;
    std::vector<bool> visited(max_node_idx, false);  // Track visited nodes
    search_node_cpu(nodes, rects, 0, query_rect, result_indices, 
                   max_node_idx, visited);
    return result_indices;
}

/**
 * @brief Test function to demonstrate quadtree functionality
 * 
 * Creates a set of test rectangles, builds a quadtree, and performs
 * intersection queries to verify correct operation.
 */
void test_quadtree() {
    // Initialize random number generator
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, 1000);
    
    // Create test rectangles
    const int NUM_RECTS = 10;
    std::vector<Rect> h_rects = {
        Rect(380, 380, 420, 420),  
        Rect(450, 450, 490, 490),  
        Rect(390, 390, 410, 410),  
        Rect(495, 495, 510, 510),  
        Rect(350, 350, 360, 360),  
        Rect(550, 550, 560, 560),  
        Rect(400, 350, 450, 480),  
        Rect(350, 420, 480, 450),  
        Rect(395, 395, 505, 505),  
        Rect(480, 380, 520, 420)   
    };
    const int NUM_TEST_RECTS = h_rects.size();

    // Assign original indices
    for (int i = 0; i < NUM_TEST_RECTS; i++) {
        h_rects[i].original_index = i;
    }

    // Allocate device memory
    Rect* d_rects = nullptr;
    Quadtree_node* d_nodes = nullptr;
    int* d_result_indices = nullptr;
    int* d_result_count = nullptr;

    // Error checking macro for CUDA calls
    #define CHECK_CUDA(call) \
        do { \
            cudaError_t error = call; \
            if (error != cudaSuccess) { \
                std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ \
                         << ": " << cudaGetErrorString(error) << std::endl; \
                return; \
            } \
        } while(0)

    // Allocate device memory
    CHECK_CUDA(cudaMalloc(&d_rects, NUM_RECTS * sizeof(Rect)));
    CHECK_CUDA(cudaMalloc(&d_nodes, NUM_RECTS * 4 * sizeof(Quadtree_node)));
    CHECK_CUDA(cudaMalloc(&d_result_indices, MAX_RESULTS * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_result_count, sizeof(int)));

    // Initialize root node
    Quadtree_node root;
    root.set_range(0, NUM_RECTS);
    
    // Copy data to device
    CHECK_CUDA(cudaMemcpy(d_nodes, &root, sizeof(Quadtree_node),
                         cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_rects, h_rects.data(), NUM_RECTS * sizeof(Rect),
                         cudaMemcpyHostToDevice));
    
    CHECK_CUDA(cudaMemset(d_result_count, 0, sizeof(int)));

    // Build quadtree
    std::cout << "Building quadtree...\n";
    const int max_depth = 18;
    int max_nodes = 0;
    for (int i = 0, num_nodes_at_level = 1; i < max_depth; 
         ++i, num_nodes_at_level *= 4) {
        max_nodes += num_nodes_at_level;
    }

    build_quadtree_kernel<<<1, 1>>>(
        d_nodes,
        d_rects,
        0,  
        0,  
        8,
        2,  
        max_nodes
    );

    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
    std::cout << "Quadtree built successfully\n";
    
    // Create test query
    Rect query_rect(400, 400, 500, 500);

    // Calculate actual number of nodes used
    const int node_buffer_size = NUM_RECTS * 4;

    // Allocate host memory for results
    std::vector<Quadtree_node> h_nodes(node_buffer_size);
    std::vector<Rect> h_search_rects(NUM_RECTS);
    
    // Copy results back to host
    CHECK_CUDA(cudaMemcpy(h_nodes.data(), d_nodes, 
                         node_buffer_size * sizeof(Quadtree_node), 
                         cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_search_rects.data(), d_rects, 
                         NUM_RECTS * sizeof(Rect), 
                         cudaMemcpyDeviceToHost));

    // Perform CPU search
    std::vector<int> cpu_results = search_rect_cpu(h_nodes.data(),
                                                  h_search_rects.data(),
                                                  query_rect,
                                                  node_buffer_size);
    
    // Output results
    std::cout << "\nFound " << cpu_results.size() << " intersecting rectangles\n";
    for (int idx : cpu_results) {
        const Rect& rect = h_rects[idx];
        std::cout << "Rectangle " << idx << ": ("
                 << rect.min_x << "," << rect.min_y << ") - ("
                 << rect.max_x << "," << rect.max_y << ")\n";
    }
    
    // Verify results
    verify_results(h_rects, cpu_results, cpu_results.size(), query_rect);

    // Clean up
    cudaFree(d_rects);
    cudaFree(d_nodes);
}

/**
 * @brief Main entry point
 */
int main() {
    test_quadtree();
    return 0;
}