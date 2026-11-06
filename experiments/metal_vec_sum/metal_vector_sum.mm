#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "metal_vector_sum.hpp"

#include <cstring>
#include <fstream>
#include <limits>
#include <sstream>

bool metal_vector_sum(
    const float* a,
    const float* b,
    float* result,
    std::size_t count,
    std::string& error) {
    @autoreleasepool {
        if (count == 0) {
            return true;
        }
        if (a == nullptr || b == nullptr || result == nullptr) {
            error = "input and output pointers must not be null";
            return false;
        }
        if (count > std::numeric_limits<NSUInteger>::max() / sizeof(float)) {
            error = "vector is too large for a Metal buffer";
            return false;
        }

        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (device == nil) {
            error = "no Metal device is available";
            return false;
        }

        std::ifstream shader_file(METAL_VEC_SUM_SHADER_PATH);
        if (!shader_file) {
            error = "could not open vector_sum.metal";
            return false;
        }
        std::ostringstream shader_stream;
        shader_stream << shader_file.rdbuf();
        NSString* shader_source = [NSString
            stringWithUTF8String:shader_stream.str().c_str()];

        NSError* ns_error = nil;
        id<MTLLibrary> library = [device newLibraryWithSource:shader_source
                                                     options:nil
                                                       error:&ns_error];
        if (library == nil) {
            error = ns_error == nil
                ? "unknown Metal error"
                : ns_error.localizedDescription.UTF8String;
            return false;
        }

        id<MTLFunction> function = [library newFunctionWithName:@"vector_sum"];
        if (function == nil) {
            error = "vector_sum kernel was not found";
            return false;
        }

        id<MTLComputePipelineState> pipeline =
            [device newComputePipelineStateWithFunction:function error:&ns_error];
        if (pipeline == nil) {
            error = ns_error == nil
                ? "unknown Metal error"
                : ns_error.localizedDescription.UTF8String;
            return false;
        }

        const NSUInteger byte_count = count * sizeof(float);
        id<MTLBuffer> buffer_a = [device newBufferWithBytes:a
                                                    length:byte_count
                                                   options:MTLResourceStorageModeShared];
        id<MTLBuffer> buffer_b = [device newBufferWithBytes:b
                                                    length:byte_count
                                                   options:MTLResourceStorageModeShared];
        id<MTLBuffer> buffer_result = [device newBufferWithLength:byte_count
                                                         options:MTLResourceStorageModeShared];
        if (buffer_a == nil || buffer_b == nil || buffer_result == nil) {
            error = "failed to allocate Metal buffers";
            return false;
        }

        id<MTLCommandQueue> queue = [device newCommandQueue];
        id<MTLCommandBuffer> command_buffer = [queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
        if (queue == nil || command_buffer == nil || encoder == nil) {
            error = "failed to create Metal command objects";
            return false;
        }

        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:buffer_a offset:0 atIndex:0];
        [encoder setBuffer:buffer_b offset:0 atIndex:1];
        [encoder setBuffer:buffer_result offset:0 atIndex:2];

        const NSUInteger thread_width = pipeline.threadExecutionWidth;
        const MTLSize threads_per_grid = MTLSizeMake(count, 1, 1);
        const MTLSize threads_per_group = MTLSizeMake(thread_width, 1, 1);
        [encoder dispatchThreads:threads_per_grid
           threadsPerThreadgroup:threads_per_group];
        [encoder endEncoding];

        [command_buffer commit];
        [command_buffer waitUntilCompleted];
        if (command_buffer.status == MTLCommandBufferStatusError) {
            error = command_buffer.error == nil
                ? "unknown Metal error"
                : command_buffer.error.localizedDescription.UTF8String;
            return false;
        }

        std::memcpy(result, buffer_result.contents, byte_count);
        return true;
    }
}
