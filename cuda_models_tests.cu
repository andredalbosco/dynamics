#include <Eigen/LU>
#include <type_traits>
#include "core/xpbd_gpu.hpp"
#include "core/parallel_cloth_gpu.hpp"
#include "core/mpm_gpu.hpp"
#include "tests/scene_fixtures.hpp"
#include <cstdio>
#include <string>

// Internal kernel declaration for the isolated contact projection check.
namespace falling_cuda {
__global__ void integrate(FallingDeviceParticle* p, int count, float h, float3 gravity,
    float3 sphere, float radius, float thickness, float friction, bool sphereEnabled, bool groundEnabled);
}

static void require(bool good,const char* message) {
    if(!good) throw std::runtime_error(message);
}

template<class Solver>
static void ownership(Solver first, Solver second) {
    static_assert(!std::is_copy_constructible_v<Solver>);
    static_assert(!std::is_copy_assignable_v<Solver>);
    static_assert(std::is_nothrow_move_constructible_v<Solver>);
    static_assert(std::is_nothrow_move_assignable_v<Solver>);
    auto* firstAllocation = first.deviceParticles;
    auto* secondAllocation = second.deviceParticles;
    const auto firstCount = first.particles.size(), secondCount = second.particles.size();
    Solver moved {std::move(first)};
    require(first.deviceParticles == nullptr && moved.deviceParticles == firstAllocation,
        "GPU move constructor transfers ownership");
    moved.step(0.001);
    moved = std::move(second);
    require(moved.deviceParticles == secondAllocation && second.deviceParticles == firstAllocation,
        "GPU move assignment preserves both live allocations");
    require(moved.particles.size() == secondCount && second.particles.size() == firstCount,
        "GPU move assignment transfers matching host state");
    moved.step(0.001);
    second.step(0.001);
    auto* alias = &moved;
    moved = std::move(*alias);
    require(moved.deviceParticles == secondAllocation, "GPU self move preserves allocation");
    moved.step(0.001);
    first = std::move(moved);
    require(first.deviceParticles == secondAllocation && moved.deviceParticles == nullptr,
        "GPU move assignment into moved-from solver");
    first.step(0.001);
}

static void ownershipTests() {
    ownership(makeCloth<xpbd::CudaSolver>(3,4,0.2,0.01), makeCloth<xpbd::CudaSolver>(4,5,0.2,0.01));
    ownership(CudaParallelCloth {4,6}, CudaParallelCloth {8,4});
    auto second = makeMpmScene();
    second.young = 700;
    ownership(CudaMpmSimulation {makeMpmScene()}, CudaMpmSimulation {std::move(second)});
    std::puts("GPU raw-pointer ownership: moves, live replacement, self move, destruction");
}

static void clothParity(bool quick) {
    float maxPosition=0,maxVelocity=0,maxNormal=0;
    for(int mode=0;mode<4;++mode) {
        ParallelCloth cpu(8,6,0.05f,1.98f);
        CudaParallelCloth gpu(8,6,0.05f,1.98f);
        cpu.mode=gpu.mode=static_cast<ParallelCloth::Mode>(mode);
        cpu.substeps=gpu.substeps=quick?2:8;
        cpu.particles[3].invMass=0;
        cpu.particles[12].velocity={0.1f,-0.2f,0.05f};
        for(int frame=0;frame<(quick?1:30);++frame) {
            if(frame==4) cpu.particles[3].pos+=ParallelCloth::Vec(0.02f,0.01f,0);
            if(frame==6) cpu.particles[3].invMass=1;
            if(frame==9) cpu.jacobiScale=gpu.jacobiScale=0.25f;
            if(frame==12) cpu.friction=gpu.friction=0.1f;
            if(frame==15) cpu.groundEnabled=gpu.groundEnabled=false;
            // Compare complete GPU steps from identical CPU inputs.
            gpu.particles=cpu.particles;
            gpu.step(1.0f/60); cpu.step(1.0f/60);
            cpu.updateNormals(); gpu.updateNormals();
            for(size_t i=0;i<cpu.particles.size();++i) {
                const auto& a=cpu.particles[i]; const auto& b=gpu.particles[i];
                require(b.pos.allFinite()&&b.velocity.allFinite()&&gpu.normals[i].allFinite(),"Nonfinite CUDA cloth");
                maxPosition=std::max(maxPosition,(a.pos-b.pos).norm());
                maxVelocity=std::max(maxVelocity,(a.velocity-b.velocity).norm());
                maxNormal=std::max(maxNormal,(cpu.normals[i]-gpu.normals[i]).norm());
                require(a.invMass==b.invMass,"CUDA cloth changed mass");
            }
        }
        gpu.reset(); gpu.updateNormals();
        require((gpu.particles[0].pos-gpu.restPositions[0]).norm()==0,"CUDA cloth reset");
    }
    std::printf("CUDA cloth timestep parity: position %.9g, velocity %.9g, normal %.9g\n",maxPosition,maxVelocity,maxNormal);
    require(maxPosition<2.5e-5f&&maxVelocity<0.01f&&maxNormal<5e-4f,"CUDA cloth differs from CPU schedule");
    // Native contact kernel: isolate projection from later constraint corrections.
    CudaParallelCloth c(2,2);
    c.particles[0].pos={0.1f,1.5f,0}; c.particles[0].velocity.setZero();
    c.particles[1].pos={0,-1,0}; c.particles[1].velocity={1,0,0};
    c.uploadParticles();
    falling_cuda::integrate<<<1,128>>>(c.deviceParticles,int(c.particles.size()),0.1f,
        make_float3(0,0,0),make_float3(0,1.5f,0),0.5f,0.001f,0.25f,true,true);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(c.transfer.data(),c.deviceParticles,c.transfer.size()*sizeof(FallingDeviceParticle),cudaMemcpyDeviceToHost));
    require(std::abs(c.transfer[0].pos.x-0.501f)<1e-6f,"CUDA sphere thickness");
    require(std::abs(c.transfer[1].pos.y-0.001f)<1e-7f&&std::abs(c.transfer[1].pos.x-0.075f)<1e-6f,"CUDA ground friction");
}

static void mpmParity(bool quick) {
    auto cpu=makeMpmScene();
    CudaMpmSimulation gpu {cpu};
    double maximum=0;
    for(int frame=0;frame<(quick?1:8);++frame) {
        if(frame==2) cpu.young=gpu.young=700;
        if(frame==4) cpu.poisson=gpu.poisson=0.3;
        gpu.particles=cpu.particles;
        double dt=quick?0.002:1.0/60;
        gpu.step(dt); cpu.step(dt);
        require(gpu.lastSubsteps==cpu.lastSubsteps,"CUDA MPM adaptive substep count differs");
        for(size_t i=0;i<cpu.particles.size();++i) {
            const auto& a=cpu.particles[i]; const auto& b=gpu.particles[i];
            maximum=std::max(maximum,(a.pos-b.pos).norm());
            maximum=std::max(maximum,(a.vel-b.vel).norm());
            maximum=std::max(maximum,(a.affine-b.affine).norm());
            maximum=std::max(maximum,(a.deformation-b.deformation).norm());
        }
        // Device P2G must conserve total mass, independent of atomic summation order.
        std::vector<MpmDeviceNode> grid(gpu.grid.size());
        CUDA_CHECK(cudaMemcpy(grid.data(),gpu.deviceGrid,grid.size()*sizeof(MpmDeviceNode),cudaMemcpyDeviceToHost));
        double gridMass=0,particleMass=0;
        for(const auto& n:grid) gridMass+=n.mass;
        for(const auto& p:gpu.particles) particleMass+=p.mass;
        require(std::abs(gridMass-particleMass)<1e-12,"CUDA MPM P2G mass conservation");
    }
    std::printf("CUDA MPM timestep parity (x,v,C,F): %.9g\n",maximum);
    require(std::isfinite(maximum)&&maximum<2e-8,"CUDA MPM differs from CPU transfer/stress reference");
    gpu= CudaMpmSimulation {makeMpmScene()};
    require(gpu.time==0,"CUDA MPM reset");
    if(!quick) {
        for(int frame=0;frame<120;++frame) {
            if(frame==60) gpu.young=550;
            gpu.step(1.0/60);
        }
        double movement=0;
        auto initial=makeMpmScene();
        for(size_t i=0;i<gpu.particles.size();++i) {
            const auto& p=gpu.particles[i];
            require(p.pos.allFinite()&&p.vel.allFinite()&&p.deformation.allFinite()&&p.deformation.determinant()>0,"CUDA MPM independent trajectory");
            require(p.pos.minCoeff()>0&&p.pos.maxCoeff()<1,"CUDA MPM wall confinement");
            movement+=(p.pos-initial.particles[i].pos).norm();
        }
        require(movement>1&&std::abs(gpu.time-2)<1e-10,"CUDA MPM progresses for two seconds");
        std::puts("CUDA MPM independent two-second trajectory: finite, non-inverted, inside walls");
    }
    bool rejected=false;
    gpu.particles[0].pos={-1,0.5};
    try { gpu.step(0.001); } catch(const std::runtime_error&) { rejected=true; }
    require(rejected,"CUDA MPM rejects invalid stencil on device");
    gpu=CudaMpmSimulation {makeMpmScene()};
    gpu.particles[0].deformation(0,0)=-1;
    rejected=false;
    try { gpu.step(0.001); } catch(const std::runtime_error&) { rejected=true; }
    require(rejected,"CUDA MPM rejects inverted deformation on device");
}

int main(int argc,char** argv) {
    try {
        const bool quick=argc==2&&std::string(argv[1])=="--quick";
        if(argc!=1&&!quick) return 2;
        ownershipTests(); clothParity(quick); mpmParity(quick);
        std::puts("CUDA model tests passed"); return 0;
    } catch(const std::exception& e) {
        std::fprintf(stderr,"CUDA model tests: %s\n",e.what()); return 1;
    }
}
