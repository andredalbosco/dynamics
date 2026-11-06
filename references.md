# References

## Papers

| Topic | Papers |
| --- | --- |
| PBD | [Position Based Dynamics](https://matthias-research.github.io/pages/publications/posBasedDyn.pdf), [survey](https://animation.rwth-aachen.de/publication/0553/) |
| XPBD | [XPBD](https://mmacklin.com/xpbd.pdf), [Small Steps](https://matthias-research.github.io/pages/publications/smallsteps.pdf) |
| PD | [Projective Dynamics](https://www.projectivedynamics.org/projectivedynamics.pdf), [Fast Mass-Spring Simulation](https://users.cs.utah.edu/~ladislav/liu13fast/liu13fast.pdf) |
| VBD | [Vertex Block Descent](https://ankachan.github.io/Projects/VertexBlockDescent/index.html) |
| AVBD | [Augmented Vertex Block Descent](https://graphics.cs.utah.edu/research/projects/avbd/) |
| Baraff–Witkin | *Large Steps in Cloth Simulation* (1998) |
| Primal/dual methods | [Primal/Dual Descent](https://mmacklin.com/primaldual.pdf), [Primal XPBD](https://physbam.stanford.edu/~fedkiw/papers/stanford2023-11.pdf), [journal extension](https://doi.org/10.1016/j.cag.2024.103902) |
| Continuum constraints | [Stable Neo-Hookean constraints](https://matthias-research.github.io/pages/publications/neohookean.pdf) |
| Multilevel XPBD | [Multi-layer XPBD](https://www.alexandremercieraubin.com/Work/papers/MultiLayerXPBD/), [MGPBD](https://arxiv.org/abs/2505.13390) |
| Conservation | [Constrained Projective Dynamics](https://doi.org/10.1145/3450626.3459878) |
| Contact | [IPC](https://ipc-sim.github.io/), [DAT](https://ankachan.github.io/Projects/DAT/) |

## Repositories

| Repository | Focus |
| --- | --- |
| [HOBAK](https://github.com/theodorekim/HOBAKv1) | Elastic energies, gradients, Hessians, implicit integration, derivative checks |
| [PositionBasedDynamics](https://github.com/InteractiveComputerGraphics/PositionBasedDynamics) | Constraints; [XPBD implementation](https://github.com/InteractiveComputerGraphics/PositionBasedDynamics/blob/master/PositionBasedDynamics/XPBD.cpp) |
| [ShapeOp](https://www.shapeop.org/) | Projective constraints and local/global solves |
| [DiffPD](https://github.com/mit-gfx/diff_pd_public) | Differentiable PD and numerical checks |
| [FastMassSpring](https://github.com/sam007961/FastMassSpring) | Compact fast mass-spring implementation |
| [TinyVBD](https://github.com/AnkaChan/TinyVBD) | Compact VBD examples |
| [Gaia](https://github.com/AnkaChan/Gaia) | VBD and deformable materials |
| [Newton VBD](https://github.com/newton-physics/newton/tree/main/newton/_src/solvers/vbd) | GPU VBD and contact |
| [AVBD 2D](https://github.com/savant117/avbd-demo2d), [AVBD 3D](https://github.com/savant117/avbd-demo3d) | Authors' AVBD demos |
| [IPC Toolkit](https://ipctk.xyz/) | Contact potentials, derivatives, friction |
| [PolyFEM](https://polyfem.github.io/), [MFEM](https://github.com/mfem/mfem), [deal.II](https://github.com/dealii/dealii) | FEM materials, assembly, nonlinear solves |
| [SOFA](https://github.com/sofa-framework/sofa) | Components and force fields |
| [Box2D-Lite](https://github.com/erincatto/box2d-lite) | Sequential impulses and warm starting |
| [Solver2D](https://github.com/erincatto/solver2d) | PGS/TGS/XPBD comparisons |
| [Box2D](https://github.com/erincatto/box2d), [Box3D](https://github.com/erincatto/box3d) | Contacts, joints, Soft Step, solver phases |
| [Newton Dynamics](https://github.com/MADEAPPS/newton-dynamics) | Custom joint rows and Jacobians |
| [Contact/friction course code](https://github.com/sheldona/contactFrictionSim) | Educational contact and friction solvers |
| [Taichi MPM](https://github.com/yuanming-hu/taichi_mpm) | Transfers, stress, deformation; [explained MLS-MPM example](https://github.com/yuanming-hu/taichi_mpm/blob/master/mls-mpm88-explained.cpp) |
| [CB-Geo MPM](https://github.com/cb-geo/mpm), [Karamelo](https://github.com/adevaucorbeil/karamelo) | C++ MPM materials and architecture |
| [ChainQueen](https://github.com/yuanming-hu/ChainQueen) | Differentiable MPM |

## Courses

| Course | Instructor(s) | Focus |
| --- | --- | --- |
| [CMU 15-763 — Physics-based Animation of Solids and Fluids (2026)](https://www.cs.cmu.edu/~15763-s26/) | Minchen Li | Solids, fluids, contact, modern simulation methods |
| [Toronto CSC417/CSC2549 — Physics-Based Animation](https://github.com/dilevin/CSC417-physics-based-animation) | David I.W. Levin | Implementation assignments, slides, recordings |
| [Cornell CS5643 — Physically Based Animation (2025)](https://www.cs.cornell.edu/courses/cs5643/2025sp/) | Steve Marschner | Integration, deformation, contact, fluids, rods |
| [Stanford CS348C — Animation and Simulation](https://graphics.stanford.edu/courses/cs348c/) | Doug James | Constraints, rigid bodies, fluids, particle-grid methods |
| [ETH — Physically-Based Simulation in Computer Graphics (2024)](https://www.vvz.ethz.ch/Vorlesungsverzeichnis/lerneinheit.view?ansicht=KATALOGDATEN&lang=en&lerneinheitId=182597&semkez=2024W) | Stelian Coros, Bernhard Thomaszewski | Continuum mechanics, solids, fluids, rods, shells; catalog entry |
| [Freiburg — Simulation in Computer Graphics](https://cg.informatik.uni-freiburg.de/teaching.htm) | Matthias Teschner | Integration, mass-spring systems, implicit solvers, SPH |
| [McGill COMP559 — Computer Animation (2026)](https://www.cs.mcgill.ca/~kry/comp559w26/) | Paul Kry | Integration, rigid bodies, contact, elasticity, fluids |
| [Waterloo CS888 — Physics-Based Animation (2016)](https://cs.uwaterloo.ca/~c2batty/courses/CS888_2016/) | Christopher Batty | Research-paper seminar: deformables, fluids, collision |
| [An Introduction to Physics-Based Animation](https://cal.cs.umbc.edu/Courses/PhysicsBasedAnimation/) | Adam W. Bargteil, Tamar Shinar | Modeling, discretization, integration, constraints |
| [Dynamic Deformables](https://www.tkim.graphics/DYNAMIC_DEFORMABLES/) | Theodore Kim, David Eberle | Elasticity, force derivatives, implicit integration, collision; HOBAK code |
| [FEM Simulation of 3D Deformable Solids](https://viterbi-web.usc.edu/~jbarbic/femdefo/) | Eftychios Sifakis, Jernej Barbič | FEM theory, discretization, model reduction |
| [The Material Point Method for Simulating Continuum Materials](https://www.math.ucla.edu/~cffjiang/research/mpmcourse/mpmcourse.pdf) | Jiang, Schroeder, Teran, Stomakhin, Selle | Continuum mechanics, transfers, constitutive models |
| [Contact and Friction Simulation for Computer Graphics](https://siggraphcontact.github.io/) | Sheldon Andrews, Kenny Erleben, Zachary Ferguson | Contact formulations, friction, numerical solvers |
| [Fluid Simulation — SIGGRAPH 2007](https://www.cs.ubc.ca/~rbridson/fluidsimulation/) | Robert Bridson, Matthias Müller-Fischer | Grid fluids, advection, pressure solves |
| [Turbulent Fluids — SIGGRAPH 2013](https://ge.in.tum.de/research/turbulent-fluids-siggraph-course/) | Nils Thuerey, Theodore Kim, Tobias Pfaff | Turbulence models, notes, slides, code |

## Books

| Book | Focus |
| --- | --- |
| [Physics-Based Simulation](https://phys-sim-book.github.io/) | Optimization-based mechanics and implementation examples |
| [Fluid Simulation for Computer Graphics — Robert Bridson](https://www.routledge.com/Fluid-Simulation-for-Computer-Graphics/Bridson/p/book/9781482232837) | Grid/particle fluids, numerical methods, free surfaces |

## Tutorials and websites

| Resource | Focus |
| --- | --- |
| [Ten Minute Physics](https://matthias-research.github.io/pages/tenMinutePhysics/index.html) | Short lessons, demos, code; #14 for XPBD cloth |
| [Physically Based Modeling — Witkin and Baraff](https://www.cs.cmu.edu/~baraff/sigcourse/) | ODEs, implicit integration, constrained and rigid-body dynamics |
| [Catto's publications](https://box2d.org/publications/) | Rigid-body constraints and contact |
| [Solver2D explanation](https://box2d.org/posts/2024/02/solver2d/) | Solver variants, relaxation, substepping |
| [Physics-Based Animation](https://www.physicsbasedanimation.com/) | Research news, papers, courses |
