#pragma once
// #include <cuda.h>
// #include <cuda_runtime.h>
// #define GLM_FORCE_CUDA
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/Sparse>
#include <array>
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <glm/glm.hpp>
#include <memory>
#include <vector>

#pragma warning(disable : 4244)
#pragma warning(disable : 4305)
using glm::dvec3;
using glm::ivec2;
using glm::ivec3;
using glm::mat3;
using glm::vec2;
using glm::vec3;
// template <typename T, typename... Args>
// T *cuda_new(Args... args)
// {
//     T *p;
//     cudaMallocManaged(&p, sizeof(T));
//     new (p) T(args...);
//     return p;
// }

// template <typename T, typename... Args>
// T *cuda_new_array(size_t n, Args... args)
// {
//     T *p;
//     cudaMallocManaged(&p, sizeof(T) * n);
//     for (size_t i = 0; i < n; i++)
//     {
//         new (p + i) T(args...);
//     }
//     return p;
// }
#define CHECK(expr)                                                                                                    \
    [&] {                                                                                                              \
        if (!(expr)) {                                                                                                 \
            fprintf(stderr, #expr " failed at %s:%d\n", __FILE__, __LINE__);                                           \
            abort();                                                                                                   \
        }                                                                                                              \
    }()
static constexpr double pi = 3.1415926535897;
class Simulation {
  public:
    const double mu0 = 1.25663706212e-16;
    struct Cell {
        static constexpr size_t max_particles = 100;
        std::array<uint32_t, max_particles> particles;
        std::atomic<uint32_t> n_particles;
        Cell() : n_particles(0) {}
    };
    struct Neighbors {
        static constexpr size_t max_neighbors = 200;
        std::array<uint32_t, max_neighbors> neighbors;
        size_t n_neighbors = 0;
    };
    // if we want to port to cuda but not want to port a std::vector (since they simply don't have __device__ attached)
    // this is our best chance
    struct Buffers {
        // SOA for max locality
        std::unique_ptr<vec3[]> particle_position;
        std::unique_ptr<vec3[]> particle_velocity;
        std::unique_ptr<vec3[]> particle_H;
        std::unique_ptr<vec3[]> particle_M;
        std::unique_ptr<vec3[]> particle_mag_moment;
        std::unique_ptr<vec3[]> particle_mag_force;
        std::unique_ptr<vec3[]> Hext;
        std::unique_ptr<float[]> density;
        std::unique_ptr<vec3[]> dvdt;
        std::unique_ptr<float[]> drhodt;
        std::unique_ptr<float[]> P;
        std::unique_ptr<Cell[]> grid;
        std::unique_ptr<Neighbors[]> neighbors;
        size_t num_particles = 0;
    };
    struct Pointers {
        vec3 *particle_position = nullptr;
        vec3 *particle_velocity = nullptr;
        vec3 *particle_H = nullptr;
        vec3 *particle_M = nullptr;
        vec3 *particle_mag_moment = nullptr;
        vec3 *particle_mag_force = nullptr;
        vec3 *Hext = nullptr;
        float *density = nullptr;
        vec3 *dvdt = nullptr;
        float *drhodt = nullptr;
        float *P = nullptr;
        Cell *grid = nullptr;
        Neighbors *neighbors = nullptr;
    };
    void init();
    float radius = 0.02f;
    float dh = radius * 1.3f;
    float c0 = 20;
    float rho0 = 1000;
    float gamma = 7;
    float kappa = 1.0;
    float alpha = 0.5;
    float dt = 0.0003;
    int size = 0;
    float mass = 0.0;
    float h = 2 * radius;       // kernel size
    float susceptibility = 0.8; // material susceptibility
    float Gamma = pow(radius, 3) * (susceptibility / (1 + susceptibility));
    ivec3 grid_size;
    dvec3 dipole = dvec3(0.5, -0.5, 0.5);
    dvec3 m = dvec3(0, 30000, 0);
    uint32_t get_index_i(const ivec3 &p) const { return p.x + p.y * grid_size.x + p.z * grid_size.x * grid_size.y; }
    ivec3 get_cell(const vec3 &p) const {
        ivec3 ip = p * vec3(grid_size);
        ip = glm::max(glm::min(ip, grid_size - 1), ivec3(0));
        return ip;
    }
    uint32_t get_grid_index(const vec3 &p) const {
        auto ip = get_cell(p);
        return get_index_i(ip);
    }
    size_t n_iter = 0;
    size_t num_particles = 4000;
    void build_grid();
    void find_neighbors();
    vec3 dvdt_momentum_term(size_t id);
    vec3 dvdt_viscosity_term(size_t id);
    vec3 dvdt_tension_term(size_t id);
    vec3 dvdt_full(size_t id);
    float drhodt(size_t id);
    void naive_collison_handling();
    float P(size_t id);
    vec3 H(vec3 r, vec3 m);
    Eigen::Matrix3d H_mat(vec3 r, vec3 m);
    float W_avr(vec3 r);
    float W(vec3 r);
    float dWdr(vec3 r);
    void eval_Hext();
    void get_R(Eigen::Matrix3d &R, const Eigen::Vector3d &rt, const Eigen::Vector3d &rs);
    float get_C1(float q);
    float get_C2(float q);
    void get_T_hat(Eigen::Matrix3d &Ts, const Eigen::Vector3d &m_hat_s, float q);
    void get_Force_Tensor(Eigen::Matrix3d &Ts, const Eigen::Vector3d &rt, const Eigen::Vector3d &rs,
                          const Eigen::Vector3d &ms);
    void compute_m(const Eigen::VectorXd &b);
    void magnetization();
    void compute_magenetic_force();

    void run_step_euler();
    void run_step_adami();

    Buffers buffers;
    Pointers pointers;
    Simulation(const std::vector<vec3> &particles) : size(size), num_particles(particles.size()) {
        init();
        for (size_t i = 0; i < num_particles; i++) {
            pointers.particle_position[i] = particles[i];
        }
    }
    void run_step();

    void visualize_field(Eigen::MatrixXd &P, Eigen::MatrixXi &F);

    dvec3 Hext(dvec3 r);
    mat3 dHext(dvec3 r);
};