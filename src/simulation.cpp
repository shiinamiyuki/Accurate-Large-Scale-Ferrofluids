#include "simulation.h"
#include <tbb/parallel_for.h>
constexpr double pi = 3.1415926535897;
// https://github.com/erizmr/SPH_Taichi
static inline double W(double r, double h) {
    auto k = 10. / (7. * pi * h * h);
    auto q = r / h;
    auto res = 0.0;
    if (q <= 1.0)
        res = k * (1 - 1.5 * q * q + 0.75 * q * q * q);
    else if (q < 2.0) {
        auto two_m_q = 2 - q;
        res = k * 0.25 * two_m_q * two_m_q * two_m_q;
    }
    return res;
}
static inline double dW(double r, double h) {
    auto k = 10. / (7. * pi * h * h);
    auto q = r / h;
    auto res = 0.0;
    if (q <= 1.0)
        res = (k / h) * (-3 * q + 2.25 * q * q);
    else if (q < 2.0) {
        auto two_m_q = 2 - q;
        res = -0.75 * (k / h) * two_m_q * two_m_q;
    }
    return res;
}

static inline vec3 gradW(vec3 r, float h) {
    if (dot(r, r) == 0.0)
        return vec3(0.0);
    return (float)dW(length(r), h) * normalize(r);
}

void Simulation::init() {
    grid_size = floor(vec3(1) / vec3(2.0 * dh));
    buffers.num_particles = num_particles;
    buffers.particle_position.reset(new vec3[num_particles]);
    buffers.particle_velocity.reset(new vec3[num_particles]);
    buffers.density.reset(new float[num_particles]);
    buffers.drhodt.reset(new float[num_particles]);
    buffers.dvdt.reset(new vec3[num_particles]);
    buffers.grid.reset(new Cell[grid_size.x * grid_size.y * grid_size.z]);
    buffers.neighbors.reset(new Neighbors[num_particles]);
    buffers.P.reset(new float[num_particles]);
    pointers.particle_position = buffers.particle_position.get();
    pointers.particle_velocity = buffers.particle_velocity.get();
    pointers.density = buffers.density.get();
    pointers.drhodt = buffers.drhodt.get();
    pointers.dvdt = buffers.dvdt.get();
    pointers.grid = buffers.grid.get();
    pointers.neighbors = buffers.neighbors.get();
    pointers.P = buffers.P.get();
    mass = radius * radius * radius * rho0;
    tbb::parallel_for((size_t)0, num_particles, [=](size_t i) {
        pointers.density[i] = rho0;
        pointers.particle_velocity[i] = vec3(0);
        pointers.P[i] = P(i);
    });
}
void Simulation::build_grid() {
    tbb::parallel_for(0, grid_size.x * grid_size.y * grid_size.z, [=](int i) { pointers.grid[i].n_particles = 0; });
    tbb::parallel_for((size_t)0, num_particles, [=](size_t i) {
        vec3 p = pointers.particle_position[i];
        auto gid = get_grid_index(p);
        auto cnt = pointers.grid[gid].n_particles.fetch_add(1);
        if (cnt <= Cell::max_particles) {
            pointers.grid[gid].particles[cnt] = (int)i;
        }
    });
}
void Simulation::find_neighbors() {
    tbb::parallel_for((size_t)0, num_particles, [=](size_t id) {
        vec3 p = pointers.particle_position[id];
        auto &neighbors = pointers.neighbors[id];
        neighbors.n_neighbors = 0;
        auto cell_idx = get_cell(p);
        for (int dx = -1; dx <= 1; dx++) {
            for (int dy = -1; dy <= 1; dy++) {
                for (int dz = -1; dz <= 1; dz++) {
                    auto candidate_cell_idx = cell_idx + ivec3(dx, dy, dz);
                    if (glm::all(glm::greaterThanEqual(candidate_cell_idx, ivec3(0))) &&
                        glm::all(glm::lessThan(candidate_cell_idx, grid_size))) {
                        auto &cell = pointers.grid[get_index_i(candidate_cell_idx)];
                        for (size_t i = 0; i < cell.n_particles; i++) {
                            auto j = cell.particles[i];
                            if (i == j)
                                continue;
                            auto q = pointers.particle_position[j];
                            if (length(p - q) < 2.0 * dh) {
                                if (neighbors.n_neighbors < Neighbors::max_neighbors) {
                                    neighbors.neighbors[neighbors.n_neighbors++] = j;
                                }
                            }
                        }
                    }
                }
            }
        }
    });
}
vec3 Simulation::dvdt_momentum_term(size_t id) {
    const vec3 gravity(0.0, -0.0098, 0.0);
    CHECK(mass != 0.0);
    vec3 dvdt(0.0);
    auto ra = pointers.particle_position[id];
    auto &neighbors = pointers.neighbors[id];
    auto Pa = pointers.P[id];
    auto rho_a = pointers.density[id];
    for (size_t i = 0; i < neighbors.n_neighbors; i++) {
        auto b = neighbors.neighbors[i];
        auto rb = pointers.particle_position[b];
        auto Pb = pointers.P[b];
        auto rho_b = pointers.density[b];
        CHECK(rho_a != 0.0);
        dvdt += -mass * (Pa / (rho_a * rho_a) + Pb / (rho_b * rho_b)) * gradW(ra - rb, dh);
    }
    dvdt += gravity;
    CHECK(!glm::any(glm::isnan(dvdt)));
    return dvdt;
}
vec3 Simulation::dvdt_viscosity_term(size_t id) {
    constexpr float eps = 0.01f;
    vec3 dvdt(0.0);
    auto ra = pointers.particle_position[id];
    auto va = pointers.particle_velocity[id];
    auto &neighbors = pointers.neighbors[id];
    for (size_t i = 0; i < neighbors.n_neighbors; i++) {
        auto b = neighbors.neighbors[i];
        auto rb = pointers.particle_position[b];
        auto vb = pointers.particle_velocity[b];
        auto vab = va - vb;
        auto rab = ra - rb;
        if (dot(vab, rab) < 0.0) {
            auto v = -2 * alpha * dh * c0 / (pointers.density[id] + pointers.density[b]);
            auto pi_ab = -v * dot(vab, rab) / (dot(rab, rab) + eps * dh * dh);
            dvdt += mass * pi_ab * gradW(rab, dh); // minus?
        }
    }
    CHECK(!glm::any(glm::isnan(dvdt)));
    return dvdt;
}
vec3 Simulation::dvdt_full(size_t id) {
    vec3 dvdt = vec3(0.0);
    dvdt += dvdt_momentum_term(id);
    dvdt += dvdt_viscosity_term(id);
    return dvdt;
}

float Simulation::drhodt(size_t id) {
    float drhodt = 0.0;
    auto ra = pointers.particle_position[id];
    auto va = pointers.particle_velocity[id];
    auto &neighbors = pointers.neighbors[id];
    for (size_t i = 0; i < neighbors.n_neighbors; i++) {
        auto b = neighbors.neighbors[i];
        auto rb = pointers.particle_position[b];
        auto vb = pointers.particle_velocity[b];
        auto vab = va - vb;
        auto rab = ra - rb;
        drhodt += mass * dot(vab, gradW(rab, dh));
    }
    CHECK(!std::isnan(drhodt));
    return drhodt;
}
void Simulation::naive_collison_handling() {
    tbb::parallel_for(size_t(0), num_particles, [=](size_t id) {
        auto &p = pointers.particle_position[id];
        auto &v = pointers.particle_velocity[id];
        auto k = 0.1;
        for (int i = 0; i < 3; i++) {
            if (p[i] < 0.0) {
                p[i] = 0.0;
                if (v[i] < 0.0) {
                    v[i] += -(1 + k) * v[i];
                }
            }
            if (p[i] > 1.0) {
                p[i] = 1.0;
                if (v[i] > 0.0) {
                    v[i] += -(1 + k) * v[i];
                }
            }
        }
    });
}
float Simulation::P(size_t id) {
    auto B = rho0 * c0 * c0 / gamma;
    return B * (std::pow(pointers.density[id] / rho0, gamma) - 1.0);
}
void Simulation::run_step_euler() {
    build_grid();
    find_neighbors();
    tbb::parallel_for(size_t(0), num_particles, [=](size_t id) {
        vec3 dvdt = dvdt_full(id);
        pointers.drhodt[id] = drhodt(id);
        pointers.dvdt[id] = dvdt;
    });
    tbb::parallel_for(size_t(0), num_particles, [=](size_t id) {
        pointers.particle_position[id] += dt * 0.5f * pointers.particle_velocity[id];
        pointers.particle_velocity[id] += dt * 0.5f * pointers.dvdt[id];
        pointers.density[id] += dt * pointers.drhodt[id];
        pointers.P[id] = P(id);
    });
    tbb::parallel_for(size_t(0), num_particles, [=](size_t id) {
        pointers.particle_position[id] += dt * pointers.particle_velocity[id];
        pointers.particle_velocity[id] += dt * pointers.dvdt[id];
        pointers.density[id] += dt * pointers.drhodt[id];
        pointers.P[id] = P(id);
    });
    naive_collison_handling();
    // printf("step done\n");
}
void Simulation::run_step_adami() {
    build_grid();
    find_neighbors();
    tbb::parallel_for(size_t(0), num_particles, [=](size_t id) {
        vec3 dvdt = dvdt_full(id);
        pointers.dvdt[id] = dvdt; // v(t + dt/2)
    });
    tbb::parallel_for(size_t(0), num_particles, [=](size_t id) {
        pointers.particle_velocity[id] += dt * 0.5f * pointers.dvdt[id]; // v(t + dt/2)
        pointers.particle_position[id] +=
            dt * 0.5f * pointers.particle_velocity[id]; // r(t+dt/2) = r(t) + dt/2 * v(t + dt/2)
    });
    tbb::parallel_for(size_t(0), num_particles, [=](size_t id) {
        pointers.density[id] += dt * drhodt(id); // rho(t+dt) = rho(t) + dt * drhodt(t + dt/2)
    });
    tbb::parallel_for(size_t(0), num_particles, [=](size_t id) {
        pointers.particle_position[id] +=
            dt * 0.5f * pointers.particle_velocity[id]; // r(t+dt) = r(t+dt/2) + dt/2 * v(t+dt/2)
        vec3 dvdt = dvdt_full(id);
        pointers.dvdt[id] = dvdt; // Insert magnetic force here
    });
    tbb::parallel_for(size_t(0), num_particles, [=](size_t id) {
        pointers.particle_velocity[id] += dt * 0.5f * pointers.dvdt[id];
        pointers.P[id] = P(id);
    });
    naive_collison_handling();
    // printf("step done\n");
}
void Simulation::run_step() { run_step_adami(); }