#ifndef MTL_CUDA_STUBS_H
#define MTL_CUDA_STUBS_H

#include <stddef.h>
#include <string.h>
#include <stdlib.h>
#include <Metal/Metal.hpp>
#include <unordered_map>
#include <iostream>

// Per-kernel-launch tracing. Off by default (it prints ~250 lines for a 250-step
// LJ melt); build with -DUCL_METAL_DEBUG to re-enable. Real errors still use cerr.
#ifdef UCL_METAL_DEBUG
#define MTL_DBG(...) printf(__VA_ARGS__)
#else
#define MTL_DBG(...) ((void)0)
#endif

namespace ucl_metal {
    extern std::unordered_map<void*, MTL::Buffer*> ucl_metal_buffers;
}

typedef void* CUdeviceptr;
typedef MTL::CommandQueue* CUstream;
typedef int CUresult;
typedef void* CUcontext;
typedef int CUdevice;
typedef MTL::SharedEvent* CUevent;
typedef MTL::Library* CUmodule;
typedef MTL::ComputePipelineState* CUfunction;
typedef int CUmemorytype;
typedef void* CUtexref;
typedef int CUjit_option;

struct CUDA_MEMCPY2D {
    size_t srcXInBytes, srcY, srcMemoryType;
    const void* srcHost;
    void* srcDevice;
    void* srcArray;
    size_t srcPitch;
    size_t dstXInBytes, dstY, dstMemoryType;
    void* dstHost;
    void* dstDevice;
    void* dstArray;
    size_t dstPitch;
    size_t WidthInBytes, Height;
};

#define CU_COMPUTEMODE_DEFAULT 0
#define CUDA_SUCCESS 0
#define CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR 1
#define CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR 2
#define CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT 3
#define CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK 4
#define CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X 5
#define CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y 6
#define CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z 7
#define CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X 8
#define CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y 9
#define CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z 10
#define CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK 11
#define CU_DEVICE_ATTRIBUTE_TOTAL_CONSTANT_MEMORY 12
#define CU_DEVICE_ATTRIBUTE_WARP_SIZE 13
#define CU_DEVICE_ATTRIBUTE_MAX_PITCH 14
#define CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK 15
#define CU_DEVICE_ATTRIBUTE_CLOCK_RATE 16
#define CU_DEVICE_ATTRIBUTE_TEXTURE_ALIGNMENT 17
#define CU_DEVICE_ATTRIBUTE_KERNEL_EXEC_TIMEOUT 18
#define CU_DEVICE_ATTRIBUTE_INTEGRATED 0
#define CU_DEVICE_ATTRIBUTE_CAN_MAP_HOST_MEMORY 0
#define CU_DEVICE_ATTRIBUTE_COMPUTE_MODE 0
#define CU_DEVICE_ATTRIBUTE_CONCURRENT_KERNELS 0
#define CU_DEVICE_ATTRIBUTE_ECC_ENABLED 0
#define CU_COMPUTEMODE_EXCLUSIVE_PROCESS 0
#define CU_COMPUTEMODE_PROHIBITED 0

#define CU_MEMHOSTALLOC_WRITECOMBINED 0
#define CU_MEMORYTYPE_HOST 0
#define CU_MEMORYTYPE_ARRAY 1
#define CU_MEMORYTYPE_DEVICE 2

#define CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES 0
#define CU_JIT_INFO_LOG_BUFFER 0
#define CU_AD_FORMAT_FLOAT 0
#define CU_AD_FORMAT_SIGNED_INT32 0

inline int cuDeviceTotalMem(size_t* bytes, int) { 
    MTL::Device* dev = MTL::CreateSystemDefaultDevice();
    *bytes = dev ? dev->recommendedMaxWorkingSetSize() : 0;
    return 0; 
}
inline int cuDevicePrimaryCtxRetain(void**, int) { return 0; }
inline int cuDevicePrimaryCtxRelease(int) { return 0; }
inline int cuDriverGetVersion(int* v) { *v = 1000; return 0; }
inline int cuMemHostAlloc(void** ptr, size_t size, int) { *ptr = malloc(size); return 0; }

// Pre-declare ucl_metal_get_queue
inline MTL::CommandQueue* ucl_metal_get_queue(CUstream s);

inline void ucl_metal_sync_stream(CUstream s) {
    MTL::CommandQueue* q = ucl_metal_get_queue(s);
    if (q) {
        MTL::CommandBuffer* cb = q->commandBuffer();
        cb->commit();
        cb->waitUntilCompleted();
    }
}

inline int cuMemcpy2D(const CUDA_MEMCPY2D* pCopy) { 
    const void* src = pCopy->srcMemoryType == CU_MEMORYTYPE_HOST ? pCopy->srcHost : pCopy->srcDevice;
    void* dst = pCopy->dstMemoryType == CU_MEMORYTYPE_HOST ? pCopy->dstHost : pCopy->dstDevice;
    if (src && dst) {
        for (size_t y = 0; y < pCopy->Height; ++y) {
            memcpy((char*)dst + y * pCopy->dstPitch, (const char*)src + y * pCopy->srcPitch, pCopy->WidthInBytes);
        }
    }
    return 0;
}
inline int cuMemcpy2DAsync(const CUDA_MEMCPY2D* pCopy, CUstream stream) { 
    ucl_metal_sync_stream(stream);
    return cuMemcpy2D(pCopy); 
}

inline int cuModuleLoad(CUmodule* mod, const char* filename) { 
    MTL::Device* dev = MTL::CreateSystemDefaultDevice();
    NS::String* path = NS::String::string(filename, NS::UTF8StringEncoding);
    NS::Error* err = nullptr;
    *mod = dev->newLibrary(path, &err);
    if (!*mod) return 301;
    return 0; 
}

inline int cuModuleLoadDataEx(CUmodule* mod, const void* data, int, void*, void**) {
    MTL::Device* dev = MTL::CreateSystemDefaultDevice();
    MTL::CompileOptions* opts = MTL::CompileOptions::alloc()->init();
    NS::String* src = NS::String::string((const char*)data, NS::UTF8StringEncoding);
    NS::Error* err = nullptr;
    *mod = dev->newLibrary(src, opts, &err);
    opts->release();
    if (!*mod) {
        if (err) std::cerr << "Metal Compile Error: " << err->localizedDescription()->utf8String() << std::endl;
        return 301;
    }
    return 0; 
}

inline int cuModuleGetFunction(CUfunction* func, CUmodule mod, const char* name) {
    if (!mod) return 1;
    NS::String* nsName = NS::String::string(name, NS::UTF8StringEncoding);
    MTL::Function* mtlFunc = mod->newFunction(nsName);
    if (!mtlFunc) {
        NS::Array* funcs = mod->functionNames();
        for (NS::UInteger i = 0; i < funcs->count(); ++i) {
            NS::String* fname = (NS::String*)funcs->object(i);
            std::cerr << "Found function: " << fname->utf8String() << std::endl;
        }
        std::cerr << "Could not find function: " << name << std::endl;
        return 1;
    }
    MTL::Device* dev = MTL::CreateSystemDefaultDevice();
    NS::Error* err = nullptr;
    *func = dev->newComputePipelineState(mtlFunc, &err);
    mtlFunc->release();
    if (!*func) {
        if (err) std::cerr << "Failed to create pipeline: " << err->localizedDescription()->utf8String() << std::endl;
        return 1;
    }
    MTL_DBG("cuModuleGetFunction success for %s\n", name);
    fflush(stdout);
    return 0; 
}

static MTL::CommandQueue* _ucl_metal_default_queue = nullptr;
inline MTL::CommandQueue* ucl_metal_get_queue(CUstream s) {
    if (s) return s;
    if (!_ucl_metal_default_queue) {
        MTL::Device* dev = MTL::CreateSystemDefaultDevice();
        _ucl_metal_default_queue = dev->newCommandQueue();
    }
    return _ucl_metal_default_queue;
}

inline bool resolve_metal_buffer(void* val, MTL::Buffer*& out_buf, size_t& out_offset) {
    if (!val) return false;
    if (ucl_metal::ucl_metal_buffers.count(val)) {
        out_buf = ucl_metal::ucl_metal_buffers[val];
        out_offset = 0;
        return true;
    }
    for (auto const& pair : ucl_metal::ucl_metal_buffers) {
        char* base = (char*)pair.first;
        char* ptr = (char*)val;
        MTL::Buffer* buf = pair.second;
        if (ptr >= base && ptr < base + buf->length()) {
            out_buf = buf;
            out_offset = (size_t)(ptr - base);
            return true;
        }
    }
    return false;
}

inline int cuLaunchKernel(CUfunction f,
                          unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ,
                          unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ,
                          unsigned int sharedMemBytes, CUstream hStream,
                          void **kernelParams, void **extra,
                          size_t *argSizes, bool* argIsDevicePtr, int numArgs) {
    MTL::CommandQueue* q = ucl_metal_get_queue(hStream);
    if (!q) { std::cerr << "cuLaunchKernel failed: hStream is null and default queue failed" << std::endl; return 1; }
    if (!f) { std::cerr << "cuLaunchKernel failed: f is null" << std::endl; return 1; }
    MTL::CommandBuffer* cb = q->commandBuffer();
    if (!cb) { std::cerr << "cuLaunchKernel failed: cb is null" << std::endl; return 1; }
    MTL::ComputeCommandEncoder* enc = cb->computeCommandEncoder();
    if (!enc) { std::cerr << "cuLaunchKernel failed: enc is null" << std::endl; return 1; }
    enc->setComputePipelineState(f);
    
    if (sharedMemBytes > 0) {
        enc->setThreadgroupMemoryLength(sharedMemBytes, 0); 
    }

    for(int i=0; i<numArgs; ++i) {
        if (argSizes[i] == sizeof(CUdeviceptr) && argIsDevicePtr[i]) {
            CUdeviceptr ptr = *(CUdeviceptr*)kernelParams[i];
            MTL::Buffer* buf = nullptr;
            size_t offset = 0;
            if (resolve_metal_buffer(ptr, buf, offset)) {
                enc->setBuffer(buf, offset, i);
                MTL_DBG("Arg %d: IS DEVICE PTR, val=%p buf=%p offset=%zu\n", i, (void*)ptr, (void*)buf, offset);
            } else {
                enc->setBuffer(nullptr, 0, i);
                MTL_DBG("Arg %d: DEVICE PTR NOT FOUND in tracking, val=%p\n", i, (void*)ptr);
            }
        } else {
            enc->setBytes(kernelParams[i], argSizes[i], i);
            MTL_DBG("Arg %d: IS VALUE, size=%zu\n", i, argSizes[i]);
            if (argSizes[i] == 4 && i == 3) {
                MTL_DBG("GPU lj_types = %d\n", *(int*)kernelParams[i]);
            }
        }
    }
    
    // eflag, vflag, ainum, inum etc are passed by value usually in LAMMPS GPU
    // So we can assume they are properly bound via setBytes.
    
    // Get inum to determine grid size. Wait, LAMMPS cuLaunchKernel wrapper might pass grid dims.
    // In UCL_Kernel, grid and block are passed to cuLaunchKernel.
    unsigned int inum = gridDimX * blockDimX; // approximate, just for logging
    MTL_DBG("cuLaunchKernel: gridDimX=%d blockDimX=%d numArgs=%d\n", gridDimX, blockDimX, numArgs);
    MTL_DBG("GPU ainum = %d, inum = %d\n", gridDimX * blockDimX, inum);
    fflush(stdout);
    // Dispatch full, uniform threadgroups (CUDA grid-of-blocks semantics): gridDim
    // is the number of threadgroups, blockDim the threads per group. This keeps every
    // threadgroup at the full (power-of-two) block size so in-kernel threadgroup
    // reductions are correct, and makes threadgroups_per_grid == gridDimX == the host's
    // red_blocks (the stride the Answer readback assumes). Out-of-range threads
    // (thread_position_in_grid >= inum) must be guarded inside each kernel.
    MTL::Size tgroups = MTL::Size::Make(gridDimX, gridDimY, gridDimZ);
    MTL::Size block = MTL::Size::Make(blockDimX, blockDimY, blockDimZ);
    enc->dispatchThreadgroups(tgroups, block);
    enc->endEncoding();
    cb->commit();
    cb->waitUntilCompleted();
    return 0; 
}

inline int cuMemAlloc(CUdeviceptr* dptr, size_t size) { 
    MTL::Device* dev = MTL::CreateSystemDefaultDevice();
    MTL::Buffer* buf = dev->newBuffer(size, MTL::ResourceStorageModeShared);
    if (!buf) return 2;
    *dptr = buf->contents();
    ucl_metal::ucl_metal_buffers[*dptr] = buf;
    return 0; 
}

inline int cuMemFree(CUdeviceptr dptr) {
    if (dptr && ucl_metal::ucl_metal_buffers.count(dptr)) {
        ucl_metal::ucl_metal_buffers[dptr]->release();
        ucl_metal::ucl_metal_buffers.erase(dptr);
    }
    return 0; 
}
inline int cuMemcpyHtoD(void* dst, const void* src, size_t size) { ucl_metal_sync_stream(nullptr); memcpy(dst, src, size); return 0; }
inline int cuMemcpyDtoH(void* dst, const void* src, size_t size) { ucl_metal_sync_stream(nullptr); memcpy(dst, src, size); return 0; }
inline int cuMemcpyDtoD(void* dst, const void* src, size_t size) { ucl_metal_sync_stream(nullptr); memcpy(dst, src, size); return 0; }

inline int cuMemcpyHtoDAsync(void* dst, const void* src, size_t size, CUstream s) { ucl_metal_sync_stream(s); memcpy(dst, src, size); return 0; }
inline int cuMemcpyDtoHAsync(void* dst, const void* src, size_t size, CUstream s) { 
    ucl_metal_sync_stream(s); 
    memcpy(dst, src, size); 
    if (size >= 16) {
        float* f = (float*)dst;
        MTL_DBG("DtoH copy size=%zu, first 4 floats: %f %f %f %f\n", size, f[0], f[1], f[2], f[3]);
    }
    return 0; 
}
inline int cuMemcpyDtoDAsync(void* dst, const void* src, size_t size, CUstream s) { ucl_metal_sync_stream(s); memcpy(dst, src, size); return 0; }

inline int cuStreamCreate(CUstream* pStream, int) {
    MTL::Device* dev = MTL::CreateSystemDefaultDevice();
    if (!dev) {
        std::cerr << "cuStreamCreate failed: MTL::CreateSystemDefaultDevice() returned null!" << std::endl;
        *pStream = nullptr;
        return 0;
    }
    *pStream = dev->newCommandQueue();
    if (!*pStream) {
        std::cerr << "cuStreamCreate failed: newCommandQueue() returned null!" << std::endl;
    }
    return 0; 
}
inline int cuStreamSynchronize(CUstream s) { ucl_metal_sync_stream(s); return 0; }
inline int cuStreamDestroy(CUstream s) { if (s) s->release(); return 0; }

inline int cuMemsetD32Async(void* dst, int value, size_t count, CUstream s) {
    ucl_metal_sync_stream(s);
    int* d = (int*)dst;
    for(size_t i=0; i<count; ++i) d[i] = value;
    return 0;
}
inline int cuMemsetD16Async(void* dst, int value, size_t count, CUstream s) {
    ucl_metal_sync_stream(s);
    short* d = (short*)dst;
    for(size_t i=0; i<count; ++i) d[i] = value;
    return 0;
}
inline int cuMemsetD8Async(void* dst, int value, size_t count, CUstream s) {
    ucl_metal_sync_stream(s);
    memset(dst, value, count);
    return 0;
}

inline int cuEventCreate(CUevent* ev, int) {
    MTL::Device* dev = MTL::CreateSystemDefaultDevice();
    *ev = dev->newSharedEvent();
    return 0; 
}
inline int cuEventDestroy(CUevent ev) { if(ev) ev->release(); return 0; }
inline int cuEventRecord(CUevent ev, CUstream stream) { 
    if (stream && ev) {
        MTL::CommandBuffer* cb = stream->commandBuffer();
        cb->encodeSignalEvent(ev, ev->signaledValue() + 1);
        cb->commit();
    }
    return 0; 
}
inline int cuEventSynchronize(CUevent ev) { return 0; }
inline int cuEventElapsedTime(float* time, CUevent start, CUevent end) { *time = 0; return 0; }

inline int cuInit(int) { return 0; }
inline int cuMemGetInfo(size_t* free, size_t* total) { 
    MTL::Device* dev = MTL::CreateSystemDefaultDevice();
    *free = dev ? dev->recommendedMaxWorkingSetSize() : 0;
    *total = *free;
    return 0; 
}
inline int cuDeviceGetCount(int* c) { *c = 1; return 0; }
inline int cuDeviceGet(int* d, int) { *d = 0; return 0; }
inline int cuDeviceGetAttribute(int* v, int attr, int) { 
    if (attr == CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK) *v = 1024;
    else if (attr == CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X) *v = 1024;
    else if (attr == CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y) *v = 1024;
    else if (attr == CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z) *v = 64;
    else if (attr == CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X) *v = 2147483647;
    else if (attr == CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y) *v = 65535;
    else if (attr == CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z) *v = 65535;
    else if (attr == CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK) *v = 49152;
    else if (attr == CU_DEVICE_ATTRIBUTE_TOTAL_CONSTANT_MEMORY) *v = 65536;
    else if (attr == CU_DEVICE_ATTRIBUTE_WARP_SIZE) *v = 32;
    else if (attr == CU_DEVICE_ATTRIBUTE_MAX_PITCH) *v = 2147483647;
    else if (attr == CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK) *v = 65536;
    else if (attr == CU_DEVICE_ATTRIBUTE_CLOCK_RATE) *v = 1000000;
    else if (attr == CU_DEVICE_ATTRIBUTE_TEXTURE_ALIGNMENT) *v = 512;
    else if (attr == CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT) *v = 16;
    else if (attr == CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR) *v = 9;
    else if (attr == CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR) *v = 0;
    else *v = 0; 
    return 0; 
}
inline int cuDeviceGetName(char* name, int len, int) { strncpy(name, "Apple Metal", len); return 0; }
inline int cuMemAllocHost(void** ptr, size_t size) { *ptr = malloc(size); return 0; }
inline int cuMemFreeHost(void* ptr) { free(ptr); return 0; }
inline int cuCtxGetCurrent(void** ctx) { *ctx = nullptr; return 0; }
inline int cuCtxSetCurrent(void*) { return 0; }

inline int cuTexRefSetFormat(void*, int, int) { return 0; }
inline int cuModuleGetGlobal(void**, size_t*, void*, const char*) { return 0; }
inline int cuModuleGetTexRef(void**, void*, const char*) { return 0; }
inline int cuTexRefSetAddress(size_t*, void*, void*, size_t) { return 0; }

#endif
