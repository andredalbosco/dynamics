#include <cuda_runtime.h>
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#define CU(call) do { auto e=(call); if(e!=cudaSuccess) throw std::runtime_error(std::string(#call)+": "+cudaGetErrorString(e)); } while(0)
using Clock=std::chrono::steady_clock;
using V=float3;
__host__ __device__ V add(V a,V b){return make_float3(a.x+b.x,a.y+b.y,a.z+b.z);}
__host__ __device__ V sub(V a,V b){return make_float3(a.x-b.x,a.y-b.y,a.z-b.z);}
__host__ __device__ V mul(V a,float s){return make_float3(a.x*s,a.y*s,a.z*s);}
__host__ __device__ float dot(V a,V b){return a.x*b.x+a.y*b.y+a.z*b.z;}
struct Edge {int a,b; float rest,alpha;};
struct Scene {
    std::vector<V> x;
    std::vector<float> w;
    std::vector<Edge> edges;
    std::vector<int> offsets,adj,order,colors;
    int degree=0;
    double color_ms=0,adj_ms=0;
};
double ms(Clock::time_point t){return std::chrono::duration<double,std::milli>(Clock::now()-t).count();}
void preprocess(Scene& s){
    auto t=Clock::now(); int n=s.x.size(), m=s.edges.size();
    std::vector<std::vector<int>> incidence(n);
    for(int c=0;c<m;++c){incidence[s.edges[c].a].push_back(c);incidence[s.edges[c].b].push_back(c);}
    s.offsets={0}; s.adj.clear();
    for(auto& a:incidence){s.degree=std::max(s.degree,int(a.size()));s.adj.insert(s.adj.end(),a.begin(),a.end());s.offsets.push_back(s.adj.size());}
    s.adj_ms=ms(t);t=Clock::now();
    std::vector<std::vector<int>> batches;
    std::vector<std::vector<unsigned char>> used;
    for(int c=0;c<m;++c){auto e=s.edges[c];int k=0;
        while(k<int(batches.size())&&(used[k][e.a]||used[k][e.b]))++k;
        if(k==int(batches.size())){batches.emplace_back();used.emplace_back(n,0);}
        batches[k].push_back(c);used[k][e.a]=used[k][e.b]=1;
    }
    s.colors={0};s.order.clear();
    for(auto& b:batches){s.order.insert(s.order.end(),b.begin(),b.end());s.colors.push_back(s.order.size());}
    s.color_ms=ms(t);
    std::vector<int> seen(m,0);
    for(int k=1;k<int(s.colors.size());++k){std::vector<int> hit(n,0);
        for(int j=s.colors[k-1];j<s.colors[k];++j){auto e=s.edges[s.order[j]];
            if(hit[e.a]++||hit[e.b]++||seen[s.order[j]]++)throw std::runtime_error("invalid coloring");}}
    for(int v:seen)if(v!=1)throw std::runtime_error("missing edge");
}
Scene cloth(int side,std::string shape){
    Scene s;std::vector<V> rest;
    std::mt19937 rng(42);std::uniform_real_distribution<float> noise(-0.08f,0.08f);
    for(int y=0;y<side;++y)for(int x=0;x<side;++x){
        V p=make_float3(float(x),float(y),0);rest.push_back(p);
        bool pin=y==0&&(x==0||x==side-1);s.w.push_back(pin?0:1);
        if(!pin){
            if(shape=="jitter")p=add(p,make_float3(noise(rng),noise(rng),noise(rng)));
            else {float u=float(x)/(side-1),v=float(y)/(side-1);p.x+=0.06f*x*v;p.y+=0.04f*y;p.z+=0.5f*std::sin(3.14159265f*u)*std::sin(3.14159265f*v);}
        }s.x.push_back(p);
    }
    auto edge=[&](int a,int b){V d=sub(rest[b],rest[a]);s.edges.push_back({a,b,std::sqrt(dot(d,d)),0});};
    // TMP-style structural, shear, and distance-bending links, all hard.
    for(int y=0;y<side;++y)for(int x=0;x<side;++x){int a=y*side+x;
        if(x+1<side)edge(a,a+1);if(y+1<side)edge(a,a+side);
        if(x+1<side&&y+1<side){edge(a,a+side+1);edge(a+1,a+side);}
        if(x+2<side)edge(a,a+2);if(y+2<side)edge(a,a+2*side);
    }
    preprocess(s);return s;
}
__host__ __device__ V correction(V a,V b,Edge e,float wa,float wb,float lambda,float& dl){
    V d=sub(b,a);float len=sqrtf(dot(d,d)),den=wa+wb+e.alpha;
    dl=(len>1e-12f&&den>0)?(-(len-e.rest)-e.alpha*lambda)/den:0;
    return len>1e-12f?mul(d,dl/len):make_float3(0,0,0);
}
__global__ void solve(const V* x,const float* w,const Edge* edges,float* lambda,V* delta,V* perEdge,int m,float omega){
    int c=blockIdx.x*blockDim.x+threadIdx.x;if(c>=m)return;auto e=edges[c];float dl;
    V d=correction(x[e.a],x[e.b],e,w[e.a],w[e.b],lambda[c],dl);lambda[c]+=omega*dl;
    if(perEdge){perEdge[c]=d;return;}
    V a=mul(d,-w[e.a]),b=mul(d,w[e.b]);
    atomicAdd(&delta[e.a].x,a.x);atomicAdd(&delta[e.a].y,a.y);atomicAdd(&delta[e.a].z,a.z);
    atomicAdd(&delta[e.b].x,b.x);atomicAdd(&delta[e.b].y,b.y);atomicAdd(&delta[e.b].z,b.z);
}
__global__ void apply(V* x,const V* delta,int n,float omega){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)x[i]=add(x[i],mul(delta[i],omega));}
__global__ void gather(V* x,const float* w,const Edge* edges,const V* perEdge,const int* offsets,const int* adj,int n,float omega){
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;V d=make_float3(0,0,0);
    for(int j=offsets[i];j<offsets[i+1];++j){int c=adj[j];d=add(d,mul(perEdge[c],edges[c].a==i?-1.f:1.f));}
    x[i]=add(x[i],mul(d,w[i]*omega));
}
__global__ void colored(V* x,const float* w,const Edge* edges,float* lambda,int first,int count){
    int c=first+blockIdx.x*blockDim.x+threadIdx.x;if(c>=first+count)return;auto e=edges[c];float dl;
    V d=correction(x[e.a],x[e.b],e,w[e.a],w[e.b],lambda[c],dl);lambda[c]+=dl;
    x[e.a]=sub(x[e.a],mul(d,w[e.a]));x[e.b]=add(x[e.b],mul(d,w[e.b]));
}
struct Error {double sum,max;};
__global__ void residual(const V* x,const Edge* edges,const float* lambda,Error* blocks,int m){
    __shared__ double sums[256],maxs[256];int t=threadIdx.x,c=blockIdx.x*256+t;
    double r=0;if(c<m){auto e=edges[c];V d=sub(x[e.b],x[e.a]);r=(sqrt(double(dot(d,d)))-e.rest+e.alpha*lambda[c])/e.rest;}
    sums[t]=r*r;maxs[t]=fabs(r);__syncthreads();
    for(int k=128;k;k/=2){if(t<k){sums[t]+=sums[t+k];maxs[t]=fmax(maxs[t],maxs[t+k]);}__syncthreads();}
    if(t==0)blocks[blockIdx.x]={sums[0],maxs[0]};
}
template<class T>T* device(const std::vector<T>& v){T* p;CU(cudaMalloc(&p,v.size()*sizeof(T)));CU(cudaMemcpy(p,v.data(),v.size()*sizeof(T),cudaMemcpyHostToDevice));return p;}
struct GPU {
    V *x,*delta,*perEdge;float *w,*lambda;Edge* edges;int *offsets,*adj;
    Error* errors;std::vector<Error> hostErrors;std::vector<int> colors;int n,m;int method;float omega;
    cudaGraph_t graph=nullptr;cudaGraphExec_t exec=nullptr;
    GPU(const Scene& s,int mode,float relaxation):colors(s.colors),n(s.x.size()),m(s.edges.size()),method(mode),omega(relaxation){
        x=device(s.x);w=device(s.w);offsets=device(s.offsets);adj=device(s.adj);
        auto e=s.edges;if(mode==2)for(int c=0;c<m;++c)e[c]=s.edges[s.order[c]];edges=device(e);
        CU(cudaMalloc(&delta,n*sizeof(V)));CU(cudaMalloc(&perEdge,m*sizeof(V)));CU(cudaMalloc(&lambda,m*sizeof(float)));
        hostErrors.resize((m+255)/256);CU(cudaMalloc(&errors,hostErrors.size()*sizeof(Error)));reset(s);
    }
    ~GPU(){if(exec)cudaGraphExecDestroy(exec);if(graph)cudaGraphDestroy(graph);cudaFree(x);cudaFree(w);cudaFree(offsets);cudaFree(adj);cudaFree(edges);cudaFree(delta);cudaFree(perEdge);cudaFree(lambda);cudaFree(errors);}
    void reset(const Scene& s){CU(cudaMemcpy(x,s.x.data(),n*sizeof(V),cudaMemcpyHostToDevice));CU(cudaMemset(lambda,0,m*sizeof(float)));}
    void sweep(){
        if(method==2){for(int k=1;k<int(colors.size());++k){int count=colors[k]-colors[k-1];colored<<<(count+255)/256,256>>>(x,w,edges,lambda,colors[k-1],count);}}
        else if(method==0){CU(cudaMemsetAsync(delta,0,n*sizeof(V)));solve<<<(m+255)/256,256>>>(x,w,edges,lambda,delta,nullptr,m,omega);apply<<<(n+255)/256,256>>>(x,delta,n,omega);}
        else {solve<<<(m+255)/256,256>>>(x,w,edges,lambda,nullptr,perEdge,m,omega);gather<<<(n+255)/256,256>>>(x,w,edges,perEdge,offsets,adj,n,omega);}
        CU(cudaGetLastError());
    }
    void capture(){CU(cudaStreamBeginCapture(cudaStreamPerThread,cudaStreamCaptureModeGlobal));
        // Use per-thread default stream for capture and ordinary launches alike.
        sweep();CU(cudaStreamEndCapture(cudaStreamPerThread,&graph));CU(cudaGraphInstantiate(&exec,graph,0));}
    void step(bool useGraph){if(useGraph)CU(cudaGraphLaunch(exec,cudaStreamPerThread));else sweep();}
    std::pair<double,double> error(){residual<<<(m+255)/256,256>>>(x,edges,lambda,errors,m);CU(cudaMemcpy(hostErrors.data(),errors,hostErrors.size()*sizeof(Error),cudaMemcpyDeviceToHost));double sum=0,mx=0;for(auto e:hostErrors){sum+=e.sum;mx=std::max(mx,e.max);}return {std::sqrt(sum/m),mx};}
    std::vector<V> positions(){std::vector<V> v(n);CU(cudaMemcpy(v.data(),x,n*sizeof(V),cudaMemcpyDeviceToHost));return v;}
};
void require(bool b,const char* what){if(!b)throw std::runtime_error(what);}
void validate(){
    Scene s;s.x={make_float3(0,0,0),make_float3(2,0,0)};s.w={1,0.5};s.edges={{0,1,1,1}};preprocess(s);
    for(int method=0;method<3;++method){GPU g(s,method,method==2?1.f:0.5f);g.sweep();auto x=g.positions();float scale=method==2?1.f:0.5f;
        require(fabs(x[0].x-0.4f*scale)<1e-6f&&fabs(x[1].x-(2-0.2f*scale))<1e-6f,"analytic compliant unequal-mass spring");}
    s=cloth(7,"jitter");for(int c=0;c<int(s.edges.size());++c)s.edges[c].alpha=(c%3==0)?0.1f:0;
    for(int method=0;method<3;++method){GPU g(s,method,0.2f);auto x=s.x;std::vector<float> lambda(s.edges.size(),0);
        for(int iteration=0;iteration<12;++iteration){auto old=x;std::vector<V> d(x.size(),make_float3(0,0,0));
            for(int j=0;j<int(s.edges.size());++j){int c=method==2?s.order[j]:j;auto e=s.edges[c];float dl;
                V corr=correction(method==2?x[e.a]:old[e.a],method==2?x[e.b]:old[e.b],e,s.w[e.a],s.w[e.b],lambda[c],dl);
                float omega=method==2?1.f:0.2f;lambda[c]+=omega*dl;
                if(method==2){x[e.a]=sub(x[e.a],mul(corr,s.w[e.a]));x[e.b]=add(x[e.b],mul(corr,s.w[e.b]));}
                else {d[e.a]=sub(d[e.a],mul(corr,s.w[e.a]));d[e.b]=add(d[e.b],mul(corr,s.w[e.b]));}}
            if(method!=2)for(int i=0;i<int(x.size());++i)x[i]=add(x[i],mul(d[i],0.2f));g.sweep();}
        auto actual=g.positions();for(int i=0;i<int(x.size());++i){V d=sub(actual[i],x[i]);require(std::sqrt(dot(d,d))<1e-5f,"GPU/CPU iteration mismatch");if(s.w[i]==0)require(dot(sub(actual[i],s.x[i]),sub(actual[i],s.x[i]))==0,"pin moved");}
        // Captured and ordinary execution must agree as well.
        g.reset(s);g.capture();for(int i=0;i<12;++i)g.step(true);auto captured=g.positions();
        for(int i=0;i<int(x.size());++i){V d=sub(captured[i],actual[i]);require(std::sqrt(dot(d,d))<1e-5f,"graph mismatch");}
    }
    std::fprintf(stderr,"PASS analytic compliant spring, mixed-compliance CPU parity, coloring, pins, CUDA graph parity\n");
}
const char* names[]={"scatter","gather","coloring"};
int main(int argc,char** argv){try {
    validate();if(argc>1&&std::string(argv[1])=="--test")return 0;
    int side=argc>1?atoi(argv[1]):32,maxiter=argc>3?atoi(argv[3]):20000;std::string shape=argc>2?argv[2]:"jitter";
    std::string prefix=argc>4?argv[4]:"results";int repeats=argc>5?atoi(argv[5]):7;
    auto scene=cloth(side,shape);int n=scene.x.size(),m=scene.edges.size();
    cudaDeviceProp prop;CU(cudaGetDeviceProperties(&prop,0));
    std::fprintf(stderr,"GPU %s; %d particles %d constraints degree %d colors %zu; adjacency %.3f ms coloring %.3f ms\n",prop.name,n,m,scene.degree,scene.colors.size()-1,scene.adj_ms,scene.color_ms);
    std::ofstream curve(prefix+"-curve.csv"),results(prefix+"-timings.csv");
    curve<<"side,shape,method,omega,sweep,rms,max\n";
    results<<"side,shape,particles,constraints,colors,max_degree,adjacency_ms,coloring_ms,method,omega,target,sweeps,rms,max,launch,repeat,gpu_ms,wall_ms\n";
    std::vector<double> targets={1e-2,1e-3,1e-4};
    struct Run {int method;float omega;std::vector<int> counts;std::vector<std::pair<double,double>> errors;};std::vector<Run> runs;
    // Sweep relaxation instead of assuming a favorable parameter for Jacobi.
    for(int method=0;method<3;++method)for(float omega:(method==2?std::vector<float> {1}:std::vector<float> {0.2f,0.25f,0.3f,0.35f,0.4f})){
        GPU g(scene,method,omega);Run run {method,omega,std::vector<int>(targets.size(),-1),std::vector<std::pair<double,double>>(targets.size())};
        auto initial=g.error();curve<<side<<','<<shape<<','<<names[method]<<','<<omega<<",0,"<<initial.first<<','<<initial.second<<'\n';
        for(int it=1;it<=maxiter;++it){g.sweep();auto e=g.error();
            if(it<=32||it%16==0)curve<<side<<','<<shape<<','<<names[method]<<','<<omega<<','<<it<<','<<e.first<<','<<e.second<<'\n';
            for(int k=0;k<int(targets.size());++k)if(run.counts[k]<0&&e.first<=targets[k]){run.counts[k]=it;run.errors[k]=e;}
            if(!std::isfinite(e.first)||e.first>10*initial.first||run.counts.back()>0)break;
        }
        std::fprintf(stderr,"%s omega %.2f: %d / %d / %d sweeps\n",names[method],omega,run.counts[0],run.counts[1],run.counts[2]);runs.push_back(run);curve.flush();
    }
    // Shuffle all measured jobs to distribute background-load/thermal drift.
    struct Job {int run,k,repeat;bool graph;};std::vector<Job> jobs;
    for(int r=0;r<int(runs.size());++r)for(int k=0;k<int(targets.size());++k)if(runs[r].counts[k]>0)
        for(bool graph:{false,true})for(int rep=-1;rep<repeats;++rep)jobs.push_back({r,k,rep,graph});
    std::mt19937 rng(123);std::shuffle(jobs.begin(),jobs.end(),rng);
    std::vector<GPU*> solvers;for(auto r:runs){auto* g=new GPU(scene,r.method,r.omega);g->capture();solvers.push_back(g);}
    cudaEvent_t start,stop;CU(cudaEventCreate(&start));CU(cudaEventCreate(&stop));
    for(auto j:jobs){auto r=runs[j.run];auto& g=*solvers[j.run];g.reset(scene);CU(cudaDeviceSynchronize());
        auto t=Clock::now();CU(cudaEventRecord(start,cudaStreamPerThread));for(int i=0;i<r.counts[j.k];++i)g.step(j.graph);
        CU(cudaEventRecord(stop,cudaStreamPerThread));CU(cudaEventSynchronize(stop));double wall=ms(t);float gpu;CU(cudaEventElapsedTime(&gpu,start,stop));auto e=g.error();
        // Timing replay must actually meet the error target (allow a small FP32 band).
        bool met=e.first<=targets[j.k]*1.02;
        if(!met){std::fprintf(stderr,"replay missed target: %s %.1f %.3g -> %.6g\n",names[r.method],r.omega,targets[j.k],e.first);}
        if(j.repeat>=0){results<<side<<','<<shape<<','<<n<<','<<m<<','<<scene.colors.size()-1<<','<<scene.degree<<','<<scene.adj_ms<<','<<scene.color_ms<<','<<names[r.method]<<','<<r.omega<<','<<targets[j.k]<<','<<r.counts[j.k]<<','<<e.first<<','<<e.second<<','<<(j.graph?"graph":"ordinary")<<','<<j.repeat<<','<<gpu<<','<<wall<<'\n';results.flush();}
    }
    for(auto* p:solvers)delete p;CU(cudaEventDestroy(start));CU(cudaEventDestroy(stop));
    return 0;
}catch(const std::exception& e){std::fprintf(stderr,"%s\n",e.what());return 1;}}
