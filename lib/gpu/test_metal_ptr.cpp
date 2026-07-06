#define NS_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION
#include <Metal/Metal.hpp>
#include <iostream>

int main() {
    MTL::Device* device = MTL::CreateSystemDefaultDevice();
    if (!device) return 1;

    const char* shaderSrc = R"(
        #include <metal_stdlib>
        using namespace metal;
        kernel void test_ptr(
            device int* out [[buffer(0)]],
            device const int* buf1 [[buffer(1)]],
            device const int* buf2 [[buffer(2)]]
        ) {
            out[0] = (buf1 == buf2) ? 1 : 0;
        }
    )";
    NS::String* srcStr = NS::String::string(shaderSrc, NS::UTF8StringEncoding);
    NS::Error* err = nullptr;
    MTL::Library* lib = device->newLibrary(srcStr, nullptr, &err);
    if (!lib) return 1;

    MTL::Function* func = lib->newFunction(NS::String::string("test_ptr", NS::UTF8StringEncoding));
    MTL::ComputePipelineState* pso = device->newComputePipelineState(func, &err);

    MTL::Buffer* outBuf = device->newBuffer(sizeof(int), MTL::ResourceStorageModeShared);
    MTL::Buffer* testBuf = device->newBuffer(sizeof(int), MTL::ResourceStorageModeShared);

    MTL::CommandQueue* queue = device->newCommandQueue();
    MTL::CommandBuffer* cmdBuf = queue->commandBuffer();
    MTL::ComputeCommandEncoder* enc = cmdBuf->computeCommandEncoder();

    enc->setComputePipelineState(pso);
    enc->setBuffer(outBuf, 0, 0);
    enc->setBuffer(testBuf, 0, 1);
    enc->setBuffer(testBuf, 0, 2);

    MTL::Size gridSize = MTL::Size(1, 1, 1);
    MTL::Size tgSize = MTL::Size(1, 1, 1);
    enc->dispatchThreads(gridSize, tgSize);
    enc->endEncoding();
    cmdBuf->commit();
    cmdBuf->waitUntilCompleted();

    int* outData = (int*)outBuf->contents();
    std::cout << "Pointers equal? " << outData[0] << std::endl;
    return 0;
}
