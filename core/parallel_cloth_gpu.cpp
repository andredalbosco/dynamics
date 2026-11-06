#include "parallel_cloth_gpu.hpp"

#include <utility>
namespace falling_cuda {
__device__ float3 add(float3 a, float3 b) { return make_float3(a.x+b.x,a.y+b.y,a.z+b.z); }
__device__ float3 sub(float3 a, float3 b) { return make_float3(a.x-b.x,a.y-b.y,a.z-b.z); }
__device__ float3 mul(float3 a, float s) { return make_float3(a.x*s,a.y*s,a.z*s); }
__device__ float length(float3 a) { return sqrtf(a.x*a.x+a.y*a.y+a.z*a.z); }
__device__ float3 cross(float3 a, float3 b) {
    return make_float3(a.y*b.z-a.z*b.y,a.z*b.x-a.x*b.z,a.x*b.y-a.y*b.x);
}
__device__ void atomicAdd3(float3* a, float3 b) {
    atomicAdd(&a->x,b.x); atomicAdd(&a->y,b.y); atomicAdd(&a->z,b.z);
}
__global__ void integrate(FallingDeviceParticle* p, int count, float h, float3 gravity,
    float3 sphere, float radius, float thickness, float friction, bool sphereEnabled, bool groundEnabled) {
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=count) return;
    auto& a=p[i]; a.previous=a.pos; if(a.invMass==0) return;
    a.velocity=add(a.velocity,mul(gravity,h)); a.pos=add(a.pos,mul(a.velocity,h));
    if(sphereEnabled && length(sub(a.pos,sphere))<radius+thickness) {
        float3 r=sub(add(mul(a.pos,1-friction),mul(a.previous,friction)),sphere);
        float d=length(r);
        if(d>1e-8f) a.pos=add(sphere,mul(r,(radius+thickness)/d));
    }
    if(groundEnabled && a.pos.y<thickness) {
        a.pos=add(mul(a.pos,1-friction),mul(a.previous,friction)); a.pos.y=thickness;
    }
}
__device__ void project(FallingDeviceParticle* p, ParallelCloth::Link c, float3* correction) {
    auto& a=p[c.a]; auto& b=p[c.b]; float w=a.invMass+b.invMass;
    float3 delta=sub(b.pos,a.pos); float d=length(delta);
    if(w==0 || d<=1e-8f) return;
    float3 dp=mul(mul(delta,1/d),(d-c.rest)/w);
    if(correction) {
        atomicAdd3(correction+c.a,mul(dp,a.invMass)); atomicAdd3(correction+c.b,mul(dp,-b.invMass));
    } else {
        a.pos=add(a.pos,mul(dp,a.invMass)); b.pos=sub(b.pos,mul(dp,b.invMass));
    }
}
__global__ void constraints(FallingDeviceParticle* p, const ParallelCloth::Link* links,
    int first, int count, float3* correction) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<count) project(p,links[first+i],correction);
}
__global__ void sequential(FallingDeviceParticle* p, const ParallelCloth::Link* links, int count) {
    if(blockIdx.x || threadIdx.x) return;
    for(int i=0;i<count;++i) project(p,links[i],nullptr);
}
__global__ void gather(const FallingDeviceParticle* p, const ParallelCloth::Link* links,
    const int* offsets, const int* incident, float3* correction, int count) {
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=count) return;
    float3 sum=make_float3(0,0,0); const auto& a=p[i];
    for(int e=offsets[i];e<offsets[i+1];++e) {
        auto c=links[incident[e]]; const auto& b=p[c.a==i?c.b:c.a];
        float w=a.invMass+b.invMass; float3 delta=sub(b.pos,a.pos); float d=length(delta);
        if(w!=0 && d>1e-8f) sum=add(sum,mul(mul(mul(delta,1/d),(d-c.rest)/w),a.invMass));
    }
    correction[i]=sum;
}
__global__ void apply(FallingDeviceParticle* p, const float3* correction, int count, float scale) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<count) p[i].pos=add(p[i].pos,mul(correction[i],scale));
}
__global__ void velocity(FallingDeviceParticle* p,int count,float h) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<count) p[i].velocity=mul(sub(p[i].pos,p[i].previous),1/h);
}
__global__ void normals(const FallingDeviceParticle* p,const int3* triangles,
    const int* offsets,const int* incident,float3* out,int count) {
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=count) return;
    float3 sum=make_float3(0,0,0);
    for(int e=offsets[i];e<offsets[i+1];++e) {
        auto t=triangles[incident[e]];
        sum=add(sum,cross(sub(p[t.y].pos,p[t.x].pos),sub(p[t.z].pos,p[t.x].pos)));
    }
    float d=length(sum); out[i]=d>1e-8f?mul(sum,1/d):make_float3(0,0,0);
}
}

CudaParallelCloth::CudaParallelCloth(int x,int y,float spacing,float height)
    : ParallelCloth(x,y,spacing,height), transfer(particles.size()), normalTransfer(particles.size()) {
    try {
        if (!particles.empty()) CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&deviceParticles), (particles.size()) * sizeof(FallingDeviceParticle)));
        if (!links.empty()) CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&deviceLinks), (links.size()) * sizeof(Link)));
        if (!particles.empty()) CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&deviceCorrections), (particles.size()) * sizeof(float3)));
        if (!particles.empty()) CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&deviceNormals), (particles.size()) * sizeof(float3)));
        if (particles.size()+1) CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&deviceOffsets), (particles.size()+1) * sizeof(int)));
        if (!links.empty()) CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&deviceIncident), (2*links.size()) * sizeof(int)));
        if (particles.size()+1) CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&deviceTriangleOffsets), (particles.size()+1) * sizeof(int)));
        if (!triangles.empty()) CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&deviceTriangleIncident), (3*triangles.size()) * sizeof(int)));
        if (!triangles.empty()) CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&deviceTriangles), (triangles.size()) * sizeof(int3)));
        CUDA_CHECK(cudaMemcpy(deviceLinks,links.data(),links.size()*sizeof(Link),cudaMemcpyHostToDevice));
        auto uploadCsr=[](const std::vector<std::vector<int>>& lists,auto& offsets,auto& indices) {
            std::vector<int> starts {0}, flat;
            for(const auto& list:lists) { flat.insert(flat.end(),list.begin(),list.end()); starts.push_back(int(flat.size())); }
            CUDA_CHECK(cudaMemcpy(offsets,starts.data(),starts.size()*sizeof(int),cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(indices,flat.data(),flat.size()*sizeof(int),cudaMemcpyHostToDevice));
        };
        uploadCsr(incident,deviceOffsets,deviceIncident);
        std::vector<int3> ts;
        std::vector<std::vector<int>> triangleIncident(particles.size());
        for(int i=0;i<int(triangles.size());++i) {
            const auto& t=triangles[i]; ts.push_back(make_int3(t[0],t[1],t[2]));
            for(int k=0;k<3;++k) triangleIncident[t[k]].push_back(i);
        }
        CUDA_CHECK(cudaMemcpy(deviceTriangles,ts.data(),ts.size()*sizeof(int3),cudaMemcpyHostToDevice));
        uploadCsr(triangleIncident,deviceTriangleOffsets,deviceTriangleIncident);
    } catch (...) {
        cudaFree(deviceParticles);
        cudaFree(deviceLinks);
        cudaFree(deviceCorrections);
        cudaFree(deviceNormals);
        cudaFree(deviceOffsets);
        cudaFree(deviceIncident);
        cudaFree(deviceTriangleOffsets);
        cudaFree(deviceTriangleIncident);
        cudaFree(deviceTriangles);
        throw;
    }
}

void CudaParallelCloth::uploadParticles() {
    if(particles.size()!=transfer.size() || links.size()!=size_t(passOffsets[5]))
        throw std::runtime_error("Recreate CUDA cloth after changing topology");
    for(size_t i=0;i<particles.size();++i) {
        const auto& p=particles[i];
        transfer[i]={make_float3(p.pos.x(),p.pos.y(),p.pos.z()),
            make_float3(p.previous.x(),p.previous.y(),p.previous.z()),
            make_float3(p.velocity.x(),p.velocity.y(),p.velocity.z()),p.invMass};
    }
    CUDA_CHECK(cudaMemcpy(deviceParticles,transfer.data(),transfer.size()*sizeof(FallingDeviceParticle),cudaMemcpyHostToDevice));
}

void CudaParallelCloth::step(float dt) {
    if(!(dt>0) || !std::isfinite(dt) || substeps<1 || !std::isfinite(jacobiScale) ||
        jacobiScale<0 || jacobiScale>1) throw std::invalid_argument("Invalid CUDA cloth timestep or relaxation");
    uploadParticles();
    int count=int(particles.size()), blocks=(count+127)/128;
    float h=dt/substeps;
    auto scatter=[&](int first,int last) {
        CUDA_CHECK(cudaMemset(deviceCorrections,0,particles.size()*sizeof(float3)));
        falling_cuda::constraints<<<(last-first+127)/128,128>>>(deviceParticles,deviceLinks,first,last-first,deviceCorrections);
        CUDA_CHECK(cudaGetLastError());
        falling_cuda::apply<<<blocks,128>>>(deviceParticles,deviceCorrections,count,jacobiScale);
        CUDA_CHECK(cudaGetLastError());
    };
    for(int s=0;s<substeps;++s) {
        falling_cuda::integrate<<<blocks,128>>>(deviceParticles,count,h,
            make_float3(gravity.x(),gravity.y(),gravity.z()),make_float3(sphereCenter.x(),sphereCenter.y(),sphereCenter.z()),
            sphereRadius,thickness,friction,sphereEnabled,groundEnabled);
        CUDA_CHECK(cudaGetLastError());
        switch(mode) {
            case Mode::Sequential:
                falling_cuda::sequential<<<1,1>>>(deviceParticles,deviceLinks,int(links.size()));
                CUDA_CHECK(cudaGetLastError()); break;
            case Mode::Hybrid:
                for(int pass=0;pass<4;++pass) {
                    int n=passOffsets[pass+1]-passOffsets[pass];
                    falling_cuda::constraints<<<(n+127)/128,128>>>(deviceParticles,deviceLinks,passOffsets[pass],n,nullptr);
                    CUDA_CHECK(cudaGetLastError());
                }
                scatter(passOffsets[4],passOffsets[5]); break;
            case Mode::Jacobi: scatter(0,int(links.size())); break;
            case Mode::Gather:
                falling_cuda::gather<<<blocks,128>>>(deviceParticles,deviceLinks,deviceOffsets,deviceIncident,deviceCorrections,count);
                CUDA_CHECK(cudaGetLastError());
                falling_cuda::apply<<<blocks,128>>>(deviceParticles,deviceCorrections,count,jacobiScale);
                CUDA_CHECK(cudaGetLastError()); break;
        }
        falling_cuda::velocity<<<blocks,128>>>(deviceParticles,count,h);
        CUDA_CHECK(cudaGetLastError());
    }
    CUDA_CHECK(cudaMemcpy(transfer.data(),deviceParticles,transfer.size()*sizeof(FallingDeviceParticle),cudaMemcpyDeviceToHost));
    for(size_t i=0;i<particles.size();++i) {
        const auto& d=transfer[i]; auto& p=particles[i];
        p.pos={d.pos.x,d.pos.y,d.pos.z}; p.previous={d.previous.x,d.previous.y,d.previous.z};
        p.velocity={d.velocity.x,d.velocity.y,d.velocity.z};
    }
}

void CudaParallelCloth::updateNormals() {
    uploadParticles();
    falling_cuda::normals<<<(int(particles.size())+127)/128,128>>>(deviceParticles,deviceTriangles,
        deviceTriangleOffsets,deviceTriangleIncident,deviceNormals,int(particles.size()));
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(normalTransfer.data(),deviceNormals,normalTransfer.size()*sizeof(float3),cudaMemcpyDeviceToHost));
    for(size_t i=0;i<normals.size();++i) normals[i]={normalTransfer[i].x,normalTransfer[i].y,normalTransfer[i].z};
}

CudaParallelCloth::~CudaParallelCloth() {
    cudaFree(deviceParticles);
    cudaFree(deviceLinks);
    cudaFree(deviceCorrections);
    cudaFree(deviceNormals);
    cudaFree(deviceOffsets);
    cudaFree(deviceIncident);
    cudaFree(deviceTriangleOffsets);
    cudaFree(deviceTriangleIncident);
    cudaFree(deviceTriangles);
}

CudaParallelCloth::CudaParallelCloth(CudaParallelCloth&& other) noexcept
    : ParallelCloth(std::move(other)),
      deviceParticles(std::exchange(other.deviceParticles, nullptr)),
      deviceLinks(std::exchange(other.deviceLinks, nullptr)),
      deviceCorrections(std::exchange(other.deviceCorrections, nullptr)),
      deviceNormals(std::exchange(other.deviceNormals, nullptr)),
      deviceOffsets(std::exchange(other.deviceOffsets, nullptr)),
      deviceIncident(std::exchange(other.deviceIncident, nullptr)),
      deviceTriangleOffsets(std::exchange(other.deviceTriangleOffsets, nullptr)),
      deviceTriangleIncident(std::exchange(other.deviceTriangleIncident, nullptr)),
      deviceTriangles(std::exchange(other.deviceTriangles, nullptr)),
      transfer(std::move(other.transfer)),
      normalTransfer(std::move(other.normalTransfer)) {}

CudaParallelCloth& CudaParallelCloth::operator=(CudaParallelCloth&& other) noexcept {
    if (this != &other) {
        std::swap(static_cast<ParallelCloth&>(*this), static_cast<ParallelCloth&>(other));
        std::swap(deviceParticles, other.deviceParticles);
        std::swap(deviceLinks, other.deviceLinks);
        std::swap(deviceCorrections, other.deviceCorrections);
        std::swap(deviceNormals, other.deviceNormals);
        std::swap(deviceOffsets, other.deviceOffsets);
        std::swap(deviceIncident, other.deviceIncident);
        std::swap(deviceTriangleOffsets, other.deviceTriangleOffsets);
        std::swap(deviceTriangleIncident, other.deviceTriangleIncident);
        std::swap(deviceTriangles, other.deviceTriangles);
        std::swap(transfer, other.transfer);
        std::swap(normalTransfer, other.normalTransfer);
    }
    return *this;
}
