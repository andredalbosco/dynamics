#include "mpm_gpu.hpp"

#include <utility>
namespace mpm_cuda {
constexpr int resolution=64, side=65;
constexpr double dx=1.0/resolution;

__device__ bool valid(const MpmDeviceParticle& p) {
    if(!isfinite(p.x)||!isfinite(p.y)||!isfinite(p.vx)||!isfinite(p.vy)) return false;
    for(int k=0;k<4;++k) if(!isfinite(p.C[k])||!isfinite(p.F[k])) return false;
    return p.F[0]*p.F[3]-p.F[1]*p.F[2]>0;
}
__global__ void statistics(const MpmDeviceParticle* p,int count,MpmDeviceStats* stats) {
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=count) return;
    const auto& a=p[i];
    if(!valid(a)) { atomicExch(&stats->invalid,1); return; }
    double cc=0,ff=0;
    for(int k=0;k<4;++k) { cc+=a.C[k]*a.C[k]; ff+=a.F[k]*a.F[k]; }
    double speed=sqrt(a.vx*a.vx+a.vy*a.vy)+dx*sqrt(cc), deformation=sqrt(ff);
    // Positive finite doubles have the same order as their unsigned bit patterns.
    atomicMax(reinterpret_cast<unsigned long long*>(&stats->speed),static_cast<unsigned long long>(__double_as_longlong(speed)));
    atomicMax(reinterpret_cast<unsigned long long*>(&stats->deformation),static_cast<unsigned long long>(__double_as_longlong(deformation)));
}
__device__ bool stencil(const MpmDeviceParticle& p,int& bx,int& by,double& fx,double& fy,
    double* wx,double* wy,MpmDeviceStats* stats) {
    if(!valid(p)) { atomicExch(&stats->invalid,1); return false; }
    double baseX=floor(p.x/dx-0.5), baseY=floor(p.y/dx-0.5);
    if(baseX<0||baseY<0||baseX+2>resolution||baseY+2>resolution) {
        atomicExch(&stats->invalid,2); return false;
    }
    bx=int(baseX); by=int(baseY); fx=p.x/dx-bx; fy=p.y/dx-by;
    wx[0]=0.5*(1.5-fx)*(1.5-fx); wx[1]=0.75-(fx-1)*(fx-1); wx[2]=0.5*(fx-0.5)*(fx-0.5);
    wy[0]=0.5*(1.5-fy)*(1.5-fy); wy[1]=0.75-(fy-1)*(fy-1); wy[2]=0.5*(fy-0.5)*(fy-0.5);
    return true;
}
__global__ void particleToGrid(const MpmDeviceParticle* particles,int count,MpmDeviceNode* grid,
    double h,double mu,double lambda,MpmDeviceStats* stats) {
    int id=blockIdx.x*blockDim.x+threadIdx.x; if(id>=count) return;
    const auto& p=particles[id];
    int bx,by; double fx,fy,wx[3],wy[3];
    if(!stencil(p,bx,by,fx,fy,wx,wy,stats)) return;
    const auto* F=p.F;
    double a=F[0]+F[3], b=F[2]-F[1], norm=hypot(a,b);
    if(!(norm>1e-14)) { atomicExch(&stats->invalid,3); return; }
    a/=norm; b/=norm;
    const double J=F[0]*F[3]-F[1]*F[2], volumeTerm=lambda*J*(J-1);
    double stress[4]={
        2*mu*((F[0]-a)*F[0]+(F[1]+b)*F[1])+volumeTerm,
        2*mu*((F[0]-a)*F[2]+(F[1]+b)*F[3]),
        2*mu*((F[2]-b)*F[0]+(F[3]-a)*F[1]),
        2*mu*((F[2]-b)*F[2]+(F[3]-a)*F[3])+volumeTerm};
    double affine[4];
    for(int k=0;k<4;++k) affine[k]=p.mass*p.C[k]-h*p.volume*4/(dx*dx)*stress[k];
    for(int i=0;i<3;++i) for(int j=0;j<3;++j) {
        double w=wx[i]*wy[j], ox=dx*(i-fx), oy=dx*(j-fy);
        auto& node=grid[(bx+i)*side+by+j];
        atomicAdd(&node.mass,w*p.mass);
        atomicAdd(&node.mx,w*(p.mass*p.vx+affine[0]*ox+affine[1]*oy));
        atomicAdd(&node.my,w*(p.mass*p.vy+affine[2]*ox+affine[3]*oy));
    }
}
__global__ void updateGrid(MpmDeviceNode* grid,double h,double gx,double gy) {
    int id=blockIdx.x*blockDim.x+threadIdx.x; if(id>=side*side) return;
    auto& n=grid[id]; if(n.mass==0) return;
    n.vx=n.mx/n.mass+h*gx; n.vy=n.my/n.mass+h*gy;
    int i=id/side,j=id%side;
    if(i<3) n.vx=fmax(0.0,n.vx); if(i>resolution-3) n.vx=fmin(0.0,n.vx);
    if(j<3) n.vy=fmax(0.0,n.vy); if(j>resolution-3) n.vy=fmin(0.0,n.vy);
}
__global__ void gridToParticle(MpmDeviceParticle* particles,int count,const MpmDeviceNode* grid,
    double h,MpmDeviceStats* stats) {
    int id=blockIdx.x*blockDim.x+threadIdx.x; if(id>=count) return;
    auto& p=particles[id]; int bx,by; double fx,fy,wx[3],wy[3];
    if(!stencil(p,bx,by,fx,fy,wx,wy,stats)) return;
    p.vx=0; p.vy=0; for(int k=0;k<4;++k) p.C[k]=0;
    for(int i=0;i<3;++i) for(int j=0;j<3;++j) {
        double w=wx[i]*wy[j],ox=dx*(i-fx),oy=dx*(j-fy),scale=4/(dx*dx)*w;
        const auto& n=grid[(bx+i)*side+by+j];
        p.vx+=w*n.vx; p.vy+=w*n.vy;
        p.C[0]+=scale*n.vx*ox; p.C[1]+=scale*n.vx*oy;
        p.C[2]+=scale*n.vy*ox; p.C[3]+=scale*n.vy*oy;
    }
    p.x+=h*p.vx; p.y+=h*p.vy;
    double f[4]; for(int k=0;k<4;++k) f[k]=p.F[k];
    p.F[0]=(1+h*p.C[0])*f[0]+h*p.C[1]*f[2];
    p.F[1]=(1+h*p.C[0])*f[1]+h*p.C[1]*f[3];
    p.F[2]=h*p.C[2]*f[0]+(1+h*p.C[3])*f[2];
    p.F[3]=h*p.C[2]*f[1]+(1+h*p.C[3])*f[3];
    if(!valid(p)) atomicExch(&stats->invalid,1);
}
}

CudaMpmSimulation::CudaMpmSimulation(MpmSimulation state)
    : MpmSimulation(std::move(state)), transfer(particles.size()) {
    try {
        if (!particles.empty()) CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&deviceParticles), (particles.size()) * sizeof(MpmDeviceParticle)));
        if (!grid.empty()) CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&deviceGrid), (grid.size()) * sizeof(MpmDeviceNode)));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&deviceStats), (1) * sizeof(MpmDeviceStats)));
    } catch (...) {
        cudaFree(deviceParticles);
        cudaFree(deviceGrid);
        cudaFree(deviceStats);
        throw;
    }
}

void CudaMpmSimulation::step(double dt) {
    if(!(dt>0)||!std::isfinite(dt)||!(young>0)||!std::isfinite(young)||
        !std::isfinite(poisson)||poisson<0||poisson>=0.5||!(density>0)||!std::isfinite(density)||
        !(maxSubstep>0)||!std::isfinite(maxSubstep)||!gravity.allFinite())
        throw std::invalid_argument("Invalid CUDA MPM timestep or material");
    if(particles.size()!=transfer.size()||particles.empty())
        throw std::invalid_argument("Recreate CUDA MPM after changing particle count");
    for(size_t i=0;i<particles.size();++i) {
        const auto& p=particles[i]; auto& d=transfer[i];
        if(!(p.mass>0)||!std::isfinite(p.mass)||!(p.volume>0)||!std::isfinite(p.volume))
            throw std::invalid_argument("Positive finite MPM masses and volumes required");
        d.x=p.pos.x(); d.y=p.pos.y(); d.vx=p.vel.x(); d.vy=p.vel.y(); d.mass=p.mass; d.volume=p.volume;
        for(int row=0;row<2;++row) for(int col=0;col<2;++col) {
            d.C[2*row+col]=p.affine(row,col); d.F[2*row+col]=p.deformation(row,col);
        }
    }
    CUDA_CHECK(cudaMemcpy(deviceParticles,transfer.data(),transfer.size()*sizeof(MpmDeviceParticle),cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(deviceStats,0,sizeof(MpmDeviceStats)));
    const int blocks=(int(particles.size())+127)/128;
    double remaining=dt; lastSubsteps=0;
    auto stats=[&]() {
        MpmDeviceStats value {};
        CUDA_CHECK(cudaMemcpy(&value,deviceStats,sizeof(value),cudaMemcpyDeviceToHost));
        if(value.invalid) throw std::runtime_error(value.invalid==2?"CUDA MPM particle left grid stencil domain":"Invalid CUDA MPM deformation/state");
        return value;
    };
    while(remaining>1e-14) {
        mpm_cuda::statistics<<<blocks,128>>>(deviceParticles,int(particles.size()),deviceStats);
        CUDA_CHECK(cudaGetLastError());
        const auto state=stats();
        const double wave=std::sqrt((lambda()+2*mu())/density);
        const double h=std::min(remaining,std::min(maxSubstep,
            0.2*dx/(state.speed+std::max(1.0,state.deformation)*wave+1e-12)));
        if(!(h>1e-10)||++lastSubsteps>100000) throw std::runtime_error("CUDA MPM timestep collapsed");
        CUDA_CHECK(cudaMemset(deviceGrid,0,grid.size()*sizeof(MpmDeviceNode)));
        CUDA_CHECK(cudaMemset(deviceStats,0,sizeof(MpmDeviceStats)));
        mpm_cuda::particleToGrid<<<blocks,128>>>(deviceParticles,int(particles.size()),deviceGrid,h,mu(),lambda(),deviceStats);
        CUDA_CHECK(cudaGetLastError());
        mpm_cuda::updateGrid<<<(int(grid.size())+127)/128,128>>>(deviceGrid,h,gravity.x(),gravity.y());
        CUDA_CHECK(cudaGetLastError());
        mpm_cuda::gridToParticle<<<blocks,128>>>(deviceParticles,int(particles.size()),deviceGrid,h,deviceStats);
        CUDA_CHECK(cudaGetLastError());
        remaining-=h; time+=h;
    }
    stats();
    CUDA_CHECK(cudaMemcpy(transfer.data(),deviceParticles,transfer.size()*sizeof(MpmDeviceParticle),cudaMemcpyDeviceToHost));
    for(size_t i=0;i<particles.size();++i) {
        auto& p=particles[i]; const auto& d=transfer[i];
        p.pos={d.x,d.y}; p.vel={d.vx,d.vy};
        for(int row=0;row<2;++row) for(int col=0;col<2;++col) {
            p.affine(row,col)=d.C[2*row+col]; p.deformation(row,col)=d.F[2*row+col];
        }
    }
}

CudaMpmSimulation::~CudaMpmSimulation() {
    cudaFree(deviceParticles);
    cudaFree(deviceGrid);
    cudaFree(deviceStats);
}

CudaMpmSimulation::CudaMpmSimulation(CudaMpmSimulation&& other) noexcept
    : MpmSimulation(std::move(other)),
      deviceParticles(std::exchange(other.deviceParticles, nullptr)),
      deviceGrid(std::exchange(other.deviceGrid, nullptr)),
      deviceStats(std::exchange(other.deviceStats, nullptr)),
      transfer(std::move(other.transfer)) {}

CudaMpmSimulation& CudaMpmSimulation::operator=(CudaMpmSimulation&& other) noexcept {
    if (this != &other) {
        std::swap(static_cast<MpmSimulation&>(*this), static_cast<MpmSimulation&>(other));
        std::swap(deviceParticles, other.deviceParticles);
        std::swap(deviceGrid, other.deviceGrid);
        std::swap(deviceStats, other.deviceStats);
        std::swap(transfer, other.transfer);
    }
    return *this;
}
