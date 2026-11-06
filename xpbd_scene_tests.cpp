#include "core/xpbd_gpu.hpp"
#include "tests/scene_fixtures.hpp"
#include <cstdio>

static void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

int main() {
    try {
        constexpr double dt = 1.0/60;
        // Fix refinement for comparisons against continuous reference dynamics.
        auto spring = makeSpring();
        spring.substeps = 30;
        xpbd::CudaSolver oscillator {spring};
        const double equilibrium = spring.particles[0].pos.x()+spring.constraints[0].restLength;
        const double amplitude = spring.particles[1].pos.x()-equilibrium;
        const double omega = std::sqrt(spring.particles[1].invMass/spring.constraints[0].compliance);
        double springError = 0;
        for (int frame=1;frame<=120;++frame) {
            oscillator.step(dt);
            springError = std::max(springError,std::abs(oscillator.particles[1].pos.x()-
                (equilibrium+amplitude*std::cos(omega*frame*dt))));
            require((oscillator.particles[0].pos-spring.particles[0].pos).norm()==0,"Spring anchor moved");
        }
        // XPBD's implicit update damps the oscillator; compare with a bounded tolerance.
        require(springError<0.04,"Spring differs excessively from the analytic oscillator");

        auto pendulum = makePendulum();
        pendulum.substeps = 30;
        xpbd::CudaSolver bob {pendulum};
        Eigen::Vector2d reference {0.65,0};
        const auto rhs = [](const Eigen::Vector2d& state) -> Eigen::Vector2d {
            return {state.y(),-9.81*std::sin(state.x())};
        };
        double angleError = 0;
        for (int frame=0;frame<120;++frame) {
            for (int substep=0;substep<16;++substep) {
                const double h = dt/16;
                const Eigen::Vector2d a=rhs(reference), b=rhs(reference+0.5*h*a);
                const Eigen::Vector2d c=rhs(reference+0.5*h*b), d=rhs(reference+h*c);
                reference += h/6*(a+2*b+2*c+d);
            }
            bob.step(dt);
            const Eigen::Vector3d offset=bob.particles[1].pos-bob.particles[0].pos;
            angleError=std::max(angleError,std::abs(std::atan2(offset.x(),-offset.y())-reference.x()));
            require(std::abs(offset.norm()-1)<1e-12,"Pendulum link length drift");
            require((bob.particles[0].pos-pendulum.particles[0].pos).norm()==0,"Pendulum anchor moved");
        }
        require(angleError<0.02,"Pendulum differs excessively from independent RK4 dynamics");

        auto initial = makeDoublePendulum();
        initial.substeps = 30;
        xpbd::CudaSolver coupled {initial};
        double linkError = 0, motion = 0;
        for (int frame=0;frame<180;++frame) {
            coupled.step(dt);
            for (const auto& p : coupled.particles)
                require(p.pos.allFinite()&&p.prevPos.allFinite(),"Nonfinite double pendulum");
            for (const auto& c : coupled.constraints)
                linkError=std::max(linkError,std::abs((coupled.particles[c.i].pos-
                    coupled.particles[c.j].pos).norm()-c.restLength));
            motion=std::max(motion,(coupled.particles.back().pos-initial.particles.back().pos).norm());
            require((coupled.particles[0].pos-initial.particles[0].pos).norm()==0,"Double pendulum anchor moved");
        }
        require(linkError<1e-3&&motion>0.1,"Double pendulum link or motion check failed");
        std::printf("XPBD scenes: spring analytic error %.6g, pendulum RK4 angle error %.6g, "
            "double-pendulum link error %.6g\n",springError,angleError,linkError);
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr,"XPBD scene tests: %s\n",error.what());
        return 1;
    }
}
