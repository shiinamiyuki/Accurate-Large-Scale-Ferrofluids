#include "simulation.h"
#include "original.h"
#include <Eigen/Core>
#include <Eigen/Sparse>
#include <iostream>
#include <tbb/parallel_for.h>

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
    buffers.particle_H.reset(new vec3[num_particles]);
    buffers.particle_M.reset(new vec3[num_particles]);
    buffers.particle_mag_moment.reset(new vec3[num_particles]);
    buffers.particle_mag_force.reset(new vec3[num_particles]);
    buffers.Hext.reset(new vec3[num_particles]);
    pointers.particle_position = buffers.particle_position.get();
    pointers.particle_velocity = buffers.particle_velocity.get();
    pointers.density = buffers.density.get();
    pointers.drhodt = buffers.drhodt.get();
    pointers.dvdt = buffers.dvdt.get();
    pointers.grid = buffers.grid.get();
    pointers.neighbors = buffers.neighbors.get();
    pointers.P = buffers.P.get();
    pointers.particle_H = buffers.particle_H.get();
    pointers.particle_M = buffers.particle_M.get();
    pointers.particle_mag_moment = buffers.particle_mag_moment.get();
    pointers.Hext = buffers.Hext.get();
    pointers.particle_mag_force = buffers.particle_mag_force.get();
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
    const vec3 gravity(0.0, -0.98, 0.0);
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
    if (enable_gravity)
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
vec3 Simulation::dvdt_tension_term(size_t id) {
    constexpr float eps = 0.01f;
    vec3 f(0.0);
    auto ra = pointers.particle_position[id];
    auto va = pointers.particle_velocity[id];
    auto &neighbors = pointers.neighbors[id];
    const auto k = 1.0f;
    for (size_t i = 0; i < neighbors.n_neighbors; i++) {
        auto b = neighbors.neighbors[i];
        auto rb = pointers.particle_position[b];
        auto vb = pointers.particle_velocity[b];
        auto vab = va - vb;
        auto rab = ra - rb;
        if (length(rab) <= k * h) {
            f += 1000.0f * mass * mass * float(std::cos(3 * pi / (2 * k * h) * length(rab))) * rab;
        }
    }
    CHECK(!glm::any(glm::isnan(f)));
    return f / mass;
}
vec3 Simulation::dvdt_full(size_t id) {
    vec3 dvdt = vec3(0.0);
    dvdt += dvdt_momentum_term(id);
    dvdt += dvdt_viscosity_term(id);
    dvdt += dvdt_tension_term(id);
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
        auto k = 0.3;
        for (int i = 0; i < 3; i++) {
            if (p[i] < lower[i]) {
                p[i] = lower[i];
                if (v[i] < 0.0) {
                    v[i] += -(1 + k) * v[i];
                }
            }
            if (p[i] > upper[i]) {
                p[i] = upper[i];
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

Eigen::Matrix3d Simulation::H_mat(vec3 r, vec3 m) {
    Eigen::Matrix3d mat;
    mat.setZero();
    float r_norm = length(r);
    vec3 r_hat = normalize(r);
    if (r_norm == 0.0) {
        mat = Eigen::Matrix3d::Identity() * W(vec3(0)) / 3.0;
    } else {
        mat += Eigen::Matrix3d::Identity() * W_avr(r) / 3.0;
        auto k = W_avr(r) - W(r);
        Eigen::Matrix3d A;
        A << r_hat[0] * r_hat[0], r_hat[0] * r_hat[1], r_hat[0] * r_hat[2], // .
            r_hat[1] * r_hat[0], r_hat[1] * r_hat[1], r_hat[1] * r_hat[2],  // .
            r_hat[2] * r_hat[0], r_hat[2] * r_hat[1], r_hat[2] * r_hat[2];  // .
        mat += k * A;
    }
    // printf("%lf\n", mat * Eigen::Vector3d(m.x,m.y,m.z), H(r, m));
    return mat;
}
vec3 Simulation::H(vec3 r, vec3 m) {
    float r_norm = length(r);
    if (r_norm == 0.0) {
        return vec3(0);
    }
    vec3 r_hat = normalize(r);
    vec3 H_r = dot(r_hat, m) * (W_avr(r) - W(r)) * r_hat - (W_avr(r) / 3.0f) * m;
    CHECK(!glm::any(glm::isnan(H_r)));
    return H_r;
}

float Simulation::W_avr(vec3 r) {
    float r_norm = length(r);
    if (r_norm == 0.0) {
        return 0.0;
    }
    float W_r_h = 0.0;
    float q = r_norm / h;
    auto q2 = q * q;
    auto q3 = q2 * q;
    auto q4 = q2 * q2;
    auto q5 = q3 * q2;
    auto q6 = q3 * q3;
    if (0 <= q && q < 1) {
        W_r_h = (1.0 / 40.0) * (15.0 * q3 - 36 * q2 + 40.0);
    } else if (1 <= q && q < 2) {
        W_r_h = (-3.0 / (4.0 * q3)) * (q6 / 6.0 - (6.0 * q5) / 5.0 + 3.0 * pow(q, 4) - (8.0 * q3) / 3.0 + 1.0 / 15.0);
    } else {
        W_r_h = 3.0 / (4.0 * q3);
    }
    W_r_h *= (1.0 / pi);
    W_r_h *= (1.0 / (h * h * h));
    CHECK(!std::isnan(W_r_h));
    return W_r_h;
}

float Simulation::W(vec3 r) {
    float r_norm = length(r);
    float W_r_h = 0.0;
    float q = r_norm / h;
    if (0 <= q && q < 1) {
        W_r_h = 0.25 * pow((2.0 - q), 3) - pow((1.0 - q), 3);
    }
    if (1 <= q && q < 2) {
        W_r_h = 0.25 * pow((2.0 - q), 3);
    }
    W_r_h *= (1.0 / pi);
    W_r_h *= (1.0 / (h * h * h));
    CHECK(!std::isnan(W_r_h));
    return W_r_h;
}

float Simulation::dWdr(vec3 r) {
    float r_norm = length(r);
    float dW_r_h = 0.0;
    float q = r_norm / h;
    auto q2 = q * q;
    if (0 <= q && q < 1) {
        dW_r_h = 2.25 * q2 - 3.0 * q;
    }
    if (1 <= q && q < 2) {
        dW_r_h = -0.75 * q2 + 3.0 * q - 3.0;
    }
    dW_r_h *= (1.0 / pi);
    dW_r_h *= (1.0 / (h * h * h));
    return dW_r_h;
}
mat3 Simulation::dHext(dvec3 r) {
    if (dot(r, r) == 0.0) {
        return mat3(0.0);
    }
    auto m1 = m[0];
    auto m2 = m[1];
    auto m3 = m[2];
    auto r1 = r[0];
    auto r2 = r[1];
    auto r3 = r[2];
    glm::dmat3 T;
    T[0][0] = -((m1 * r1 * 1.0 / sqrt(pow(fabs(r1), 2.0) + pow(fabs(r2), 2.0) + pow(fabs(r3), 2.0)) +
                 m2 * r2 * 1.0 / sqrt(pow(fabs(r1), 2.0) + pow(fabs(r2), 2.0) + pow(fabs(r3), 2.0)) +
                 m3 * r3 * 1.0 / sqrt(pow(fabs(r1), 2.0) + pow(fabs(r2), 2.0) + pow(fabs(r3), 2.0))) *
                    1.0 / sqrt(pow(fabs(r1), 2.0) + pow(fabs(r2), 2.0) + pow(fabs(r3), 2.0)) * -3.0 +
                r1 * 1.0 / sqrt(pow(fabs(r1), 2.0) + pow(fabs(r2), 2.0) + pow(fabs(r3), 2.0)) *
                    (-m1 * 1.0 / sqrt(pow(fabs(r1), 2.0) + pow(fabs(r2), 2.0) + pow(fabs(r3), 2.0)) +
                     m1 * r1 * fabs(r1) * ((r1 / fabs(r1))) * 1.0 /
                         pow(pow(fabs(r1), 2.0) + pow(fabs(r2), 2.0) + pow(fabs(r3), 2.0), 3.0 / 2.0) +
                     m2 * r2 * fabs(r1) * ((r1 / fabs(r1))) * 1.0 /
                         pow(pow(fabs(r1), 2.0) + pow(fabs(r2), 2.0) + pow(fabs(r3), 2.0), 3.0 / 2.0) +
                     m3 * r3 * fabs(r1) * ((r1 / fabs(r1))) * 1.0 /
                         pow(pow(fabs(r1), 2.0) + pow(fabs(r2), 2.0) + pow(fabs(r3), 2.0), 3.0 / 2.0)) *
                    3.0 +
                r1 * fabs(r1) * ((r1 / fabs(r1))) *
                    (m1 * r1 * 1.0 / sqrt(pow(fabs(r1), 2.0) + pow(fabs(r2), 2.0) + pow(fabs(r3), 2.0)) +
                     m2 * r2 * 1.0 / sqrt(pow(fabs(r1), 2.0) + pow(fabs(r2), 2.0) + pow(fabs(r3), 2.0)) +
                     m3 * r3 * 1.0 / sqrt(pow(fabs(r1), 2.0) + pow(fabs(r2), 2.0) + pow(fabs(r3), 2.0))) *
                    1.0 / pow(pow(fabs(r1), 2.0) + pow(fabs(r2), 2.0) + pow(fabs(r3), 2.0), 3.0 / 2.0) * 3.0) *
                  1.0 / pow(pow(fabs(r1), 2.0) + pow(fabs(r2), 2.0) + pow(fabs(r3), 2.0), 3.0 / 2.0) +
              fabs(r1) * ((r1 / fabs(r1))) *
                  (m1 - r1 *
                            (m1 * r1 * 1.0 / sqrt(pow(fabs(r1), 2.0) + pow(fabs(r2), 2.0) + pow(fabs(r3), 2.0)) +
                             m2 * r2 * 1.0 / sqrt(pow(fabs(r1), 2.0) + pow(fabs(r2), 2.0) + pow(fabs(r3), 2.0)) +
                             m3 * r3 * 1.0 / sqrt(pow(fabs(r1), 2.0) + pow(fabs(r2), 2.0) + pow(fabs(r3), 2.0))) *
                            1.0 / sqrt(pow(fabs(r1), 2.0) + pow(fabs(r2), 2.0) + pow(fabs(r3), 2.0)) * 3.0) *
                  1.0 / pow(pow(fabs(r1), 2.0) + pow(fabs(r2), 2.0) + pow(fabs(r3), 2.0), 5.0 / 2.0) * 3.0;
    T[0][1] = -(r1 * 1.0 / sqrt(pow(fabs(r1), 2.0) + pow(fabs(r2), 2.0) + pow(fabs(r3), 2.0)) *
                    (-m2 * 1.0 / sqrt(pow(fabs(r1), 2.0) + pow(fabs(r2), 2.0) + pow(fabs(r3), 2.0)) +
                     m1 * r1 * fabs(r2) * ((r2 / fabs(r2))) * 1.0 /
                         pow(pow(fabs(r1), 2.0) + pow(fabs(r2), 2.0) + pow(fabs(r3), 2.0), 3.0 / 2.0) +
                     m2 * r2 * fabs(r2) * ((r2 / fabs(r2))) * 1.0 /
                         pow(pow(fabs(r1), 2.0) + pow(fabs(r2), 2.0) + pow(fabs(r3), 2.0), 3.0 / 2.0) +
                     m3 * r3 * fabs(r2) * ((r2 / fabs(r2))) * 1.0 /
                         pow(pow(fabs(r1), 2.0) + pow(fabs(r2), 2.0) + pow(fabs(r3), 2.0), 3.0 / 2.0)) *
                    3.0 +
                r1 * fabs(r2) * ((r2 / fabs(r2))) *
                    (m1 * r1 * 1.0 / sqrt(pow(fabs(r1), 2.0) + pow(fabs(r2), 2.0) + pow(fabs(r3), 2.0)) +
                     m2 * r2 * 1.0 / sqrt(pow(fabs(r1), 2.0) + pow(fabs(r2), 2.0) + pow(fabs(r3), 2.0)) +
                     m3 * r3 * 1.0 / sqrt(pow(fabs(r1), 2.0) + pow(fabs(r2), 2.0) + pow(fabs(r3), 2.0))) *
                    1.0 / pow(pow(fabs(r1), 2.0) + pow(fabs(r2), 2.0) + pow(fabs(r3), 2.0), 3.0 / 2.0) * 3.0) *
                  1.0 / pow(pow(fabs(r1), 2.0) + pow(fabs(r2), 2.0) + pow(fabs(r3), 2.0), 3.0 / 2.0) +
              fabs(r2) * ((r2 / fabs(r2))) *
                  (m1 - r1 *
                            (m1 * r1 * 1.0 / sqrt(pow(fabs(r1), 2.0) + pow(fabs(r2), 2.0) + pow(fabs(r3), 2.0)) +
                             m2 * r2 * 1.0 / sqrt(pow(fabs(r1), 2.0) + pow(fabs(r2), 2.0) + pow(fabs(r3), 2.0)) +
                             m3 * r3 * 1.0 / sqrt(pow(fabs(r1), 2.0) + pow(fabs(r2), 2.0) + pow(fabs(r3), 2.0))) *
                            1.0 / sqrt(pow(fabs(r1), 2.0) + pow(fabs(r2), 2.0) + pow(fabs(r3), 2.0)) * 3.0) *
                  1.0 / pow(pow(fabs(r1), 2.0) + pow(fabs(r2), 2.0) + pow(fabs(r3), 2.0), 5.0 / 2.0) * 3.0;
    T[0][2] = -(r1 * 1.0 / sqrt(pow(fabs(r1), 2.0) + pow(fabs(r2), 2.0) + pow(fabs(r3), 2.0)) *
                    (-m3 * 1.0 / sqrt(pow(fabs(r1), 2.0) + pow(fabs(r2), 2.0) + pow(fabs(r3), 2.0)) +
                     m1 * r1 * fabs(r3) * ((r3 / fabs(r3))) * 1.0 /
                         pow(pow(fabs(r1), 2.0) + pow(fabs(r2), 2.0) + pow(fabs(r3), 2.0), 3.0 / 2.0) +
                     m2 * r2 * fabs(r3) * ((r3 / fabs(r3))) * 1.0 /
                         pow(pow(fabs(r1), 2.0) + pow(fabs(r2), 2.0) + pow(fabs(r3), 2.0), 3.0 / 2.0) +
                     m3 * r3 * fabs(r3) * ((r3 / fabs(r3))) * 1.0 /
                         pow(pow(fabs(r1), 2.0) + pow(fabs(r2), 2.0) + pow(fabs(r3), 2.0), 3.0 / 2.0)) *
                    3.0 +
                r1 * fabs(r3) * ((r3 / fabs(r3))) *
                    (m1 * r1 * 1.0 / sqrt(pow(fabs(r1), 2.0) + pow(fabs(r2), 2.0) + pow(fabs(r3), 2.0)) +
                     m2 * r2 * 1.0 / sqrt(pow(fabs(r1), 2.0) + pow(fabs(r2), 2.0) + pow(fabs(r3), 2.0)) +
                     m3 * r3 * 1.0 / sqrt(pow(fabs(r1), 2.0) + pow(fabs(r2), 2.0) + pow(fabs(r3), 2.0))) *
                    1.0 / pow(pow(fabs(r1), 2.0) + pow(fabs(r2), 2.0) + pow(fabs(r3), 2.0), 3.0 / 2.0) * 3.0) *
                  1.0 / pow(pow(fabs(r1), 2.0) + pow(fabs(r2), 2.0) + pow(fabs(r3), 2.0), 3.0 / 2.0) +
              fabs(r3) * ((r3 / fabs(r3))) *
                  (m1 - r1 *
                            (m1 * r1 * 1.0 / sqrt(pow(fabs(r1), 2.0) + pow(fabs(r2), 2.0) + pow(fabs(r3), 2.0)) +
                             m2 * r2 * 1.0 / sqrt(pow(fabs(r1), 2.0) + pow(fabs(r2), 2.0) + pow(fabs(r3), 2.0)) +
                             m3 * r3 * 1.0 / sqrt(pow(fabs(r1), 2.0) + pow(fabs(r2), 2.0) + pow(fabs(r3), 2.0))) *
                            1.0 / sqrt(pow(fabs(r1), 2.0) + pow(fabs(r2), 2.0) + pow(fabs(r3), 2.0)) * 3.0) *
                  1.0 / pow(pow(fabs(r1), 2.0) + pow(fabs(r2), 2.0) + pow(fabs(r3), 2.0), 5.0 / 2.0) * 3.0;
    T[1][0] = -(r2 * 1.0 / sqrt(pow(fabs(r1), 2.0) + pow(fabs(r2), 2.0) + pow(fabs(r3), 2.0)) *
                    (-m1 * 1.0 / sqrt(pow(fabs(r1), 2.0) + pow(fabs(r2), 2.0) + pow(fabs(r3), 2.0)) +
                     m1 * r1 * fabs(r1) * ((r1 / fabs(r1))) * 1.0 /
                         pow(pow(fabs(r1), 2.0) + pow(fabs(r2), 2.0) + pow(fabs(r3), 2.0), 3.0 / 2.0) +
                     m2 * r2 * fabs(r1) * ((r1 / fabs(r1))) * 1.0 /
                         pow(pow(fabs(r1), 2.0) + pow(fabs(r2), 2.0) + pow(fabs(r3), 2.0), 3.0 / 2.0) +
                     m3 * r3 * fabs(r1) * ((r1 / fabs(r1))) * 1.0 /
                         pow(pow(fabs(r1), 2.0) + pow(fabs(r2), 2.0) + pow(fabs(r3), 2.0), 3.0 / 2.0)) *
                    3.0 +
                r2 * fabs(r1) * ((r1 / fabs(r1))) *
                    (m1 * r1 * 1.0 / sqrt(pow(fabs(r1), 2.0) + pow(fabs(r2), 2.0) + pow(fabs(r3), 2.0)) +
                     m2 * r2 * 1.0 / sqrt(pow(fabs(r1), 2.0) + pow(fabs(r2), 2.0) + pow(fabs(r3), 2.0)) +
                     m3 * r3 * 1.0 / sqrt(pow(fabs(r1), 2.0) + pow(fabs(r2), 2.0) + pow(fabs(r3), 2.0))) *
                    1.0 / pow(pow(fabs(r1), 2.0) + pow(fabs(r2), 2.0) + pow(fabs(r3), 2.0), 3.0 / 2.0) * 3.0) *
                  1.0 / pow(pow(fabs(r1), 2.0) + pow(fabs(r2), 2.0) + pow(fabs(r3), 2.0), 3.0 / 2.0) +
              fabs(r1) * ((r1 / fabs(r1))) *
                  (m2 - r2 *
                            (m1 * r1 * 1.0 / sqrt(pow(fabs(r1), 2.0) + pow(fabs(r2), 2.0) + pow(fabs(r3), 2.0)) +
                             m2 * r2 * 1.0 / sqrt(pow(fabs(r1), 2.0) + pow(fabs(r2), 2.0) + pow(fabs(r3), 2.0)) +
                             m3 * r3 * 1.0 / sqrt(pow(fabs(r1), 2.0) + pow(fabs(r2), 2.0) + pow(fabs(r3), 2.0))) *
                            1.0 / sqrt(pow(fabs(r1), 2.0) + pow(fabs(r2), 2.0) + pow(fabs(r3), 2.0)) * 3.0) *
                  1.0 / pow(pow(fabs(r1), 2.0) + pow(fabs(r2), 2.0) + pow(fabs(r3), 2.0), 5.0 / 2.0) * 3.0;
    T[1][1] = -((m1 * r1 * 1.0 / sqrt(pow(fabs(r1), 2.0) + pow(fabs(r2), 2.0) + pow(fabs(r3), 2.0)) +
                 m2 * r2 * 1.0 / sqrt(pow(fabs(r1), 2.0) + pow(fabs(r2), 2.0) + pow(fabs(r3), 2.0)) +
                 m3 * r3 * 1.0 / sqrt(pow(fabs(r1), 2.0) + pow(fabs(r2), 2.0) + pow(fabs(r3), 2.0))) *
                    1.0 / sqrt(pow(fabs(r1), 2.0) + pow(fabs(r2), 2.0) + pow(fabs(r3), 2.0)) * -3.0 +
                r2 * 1.0 / sqrt(pow(fabs(r1), 2.0) + pow(fabs(r2), 2.0) + pow(fabs(r3), 2.0)) *
                    (-m2 * 1.0 / sqrt(pow(fabs(r1), 2.0) + pow(fabs(r2), 2.0) + pow(fabs(r3), 2.0)) +
                     m1 * r1 * fabs(r2) * ((r2 / fabs(r2))) * 1.0 /
                         pow(pow(fabs(r1), 2.0) + pow(fabs(r2), 2.0) + pow(fabs(r3), 2.0), 3.0 / 2.0) +
                     m2 * r2 * fabs(r2) * ((r2 / fabs(r2))) * 1.0 /
                         pow(pow(fabs(r1), 2.0) + pow(fabs(r2), 2.0) + pow(fabs(r3), 2.0), 3.0 / 2.0) +
                     m3 * r3 * fabs(r2) * ((r2 / fabs(r2))) * 1.0 /
                         pow(pow(fabs(r1), 2.0) + pow(fabs(r2), 2.0) + pow(fabs(r3), 2.0), 3.0 / 2.0)) *
                    3.0 +
                r2 * fabs(r2) * ((r2 / fabs(r2))) *
                    (m1 * r1 * 1.0 / sqrt(pow(fabs(r1), 2.0) + pow(fabs(r2), 2.0) + pow(fabs(r3), 2.0)) +
                     m2 * r2 * 1.0 / sqrt(pow(fabs(r1), 2.0) + pow(fabs(r2), 2.0) + pow(fabs(r3), 2.0)) +
                     m3 * r3 * 1.0 / sqrt(pow(fabs(r1), 2.0) + pow(fabs(r2), 2.0) + pow(fabs(r3), 2.0))) *
                    1.0 / pow(pow(fabs(r1), 2.0) + pow(fabs(r2), 2.0) + pow(fabs(r3), 2.0), 3.0 / 2.0) * 3.0) *
                  1.0 / pow(pow(fabs(r1), 2.0) + pow(fabs(r2), 2.0) + pow(fabs(r3), 2.0), 3.0 / 2.0) +
              fabs(r2) * ((r2 / fabs(r2))) *
                  (m2 - r2 *
                            (m1 * r1 * 1.0 / sqrt(pow(fabs(r1), 2.0) + pow(fabs(r2), 2.0) + pow(fabs(r3), 2.0)) +
                             m2 * r2 * 1.0 / sqrt(pow(fabs(r1), 2.0) + pow(fabs(r2), 2.0) + pow(fabs(r3), 2.0)) +
                             m3 * r3 * 1.0 / sqrt(pow(fabs(r1), 2.0) + pow(fabs(r2), 2.0) + pow(fabs(r3), 2.0))) *
                            1.0 / sqrt(pow(fabs(r1), 2.0) + pow(fabs(r2), 2.0) + pow(fabs(r3), 2.0)) * 3.0) *
                  1.0 / pow(pow(fabs(r1), 2.0) + pow(fabs(r2), 2.0) + pow(fabs(r3), 2.0), 5.0 / 2.0) * 3.0;
    T[1][2] = -(r2 * 1.0 / sqrt(pow(fabs(r1), 2.0) + pow(fabs(r2), 2.0) + pow(fabs(r3), 2.0)) *
                    (-m3 * 1.0 / sqrt(pow(fabs(r1), 2.0) + pow(fabs(r2), 2.0) + pow(fabs(r3), 2.0)) +
                     m1 * r1 * fabs(r3) * ((r3 / fabs(r3))) * 1.0 /
                         pow(pow(fabs(r1), 2.0) + pow(fabs(r2), 2.0) + pow(fabs(r3), 2.0), 3.0 / 2.0) +
                     m2 * r2 * fabs(r3) * ((r3 / fabs(r3))) * 1.0 /
                         pow(pow(fabs(r1), 2.0) + pow(fabs(r2), 2.0) + pow(fabs(r3), 2.0), 3.0 / 2.0) +
                     m3 * r3 * fabs(r3) * ((r3 / fabs(r3))) * 1.0 /
                         pow(pow(fabs(r1), 2.0) + pow(fabs(r2), 2.0) + pow(fabs(r3), 2.0), 3.0 / 2.0)) *
                    3.0 +
                r2 * fabs(r3) * ((r3 / fabs(r3))) *
                    (m1 * r1 * 1.0 / sqrt(pow(fabs(r1), 2.0) + pow(fabs(r2), 2.0) + pow(fabs(r3), 2.0)) +
                     m2 * r2 * 1.0 / sqrt(pow(fabs(r1), 2.0) + pow(fabs(r2), 2.0) + pow(fabs(r3), 2.0)) +
                     m3 * r3 * 1.0 / sqrt(pow(fabs(r1), 2.0) + pow(fabs(r2), 2.0) + pow(fabs(r3), 2.0))) *
                    1.0 / pow(pow(fabs(r1), 2.0) + pow(fabs(r2), 2.0) + pow(fabs(r3), 2.0), 3.0 / 2.0) * 3.0) *
                  1.0 / pow(pow(fabs(r1), 2.0) + pow(fabs(r2), 2.0) + pow(fabs(r3), 2.0), 3.0 / 2.0) +
              fabs(r3) * ((r3 / fabs(r3))) *
                  (m2 - r2 *
                            (m1 * r1 * 1.0 / sqrt(pow(fabs(r1), 2.0) + pow(fabs(r2), 2.0) + pow(fabs(r3), 2.0)) +
                             m2 * r2 * 1.0 / sqrt(pow(fabs(r1), 2.0) + pow(fabs(r2), 2.0) + pow(fabs(r3), 2.0)) +
                             m3 * r3 * 1.0 / sqrt(pow(fabs(r1), 2.0) + pow(fabs(r2), 2.0) + pow(fabs(r3), 2.0))) *
                            1.0 / sqrt(pow(fabs(r1), 2.0) + pow(fabs(r2), 2.0) + pow(fabs(r3), 2.0)) * 3.0) *
                  1.0 / pow(pow(fabs(r1), 2.0) + pow(fabs(r2), 2.0) + pow(fabs(r3), 2.0), 5.0 / 2.0) * 3.0;
    T[2][0] = -(r3 * 1.0 / sqrt(pow(fabs(r1), 2.0) + pow(fabs(r2), 2.0) + pow(fabs(r3), 2.0)) *
                    (-m1 * 1.0 / sqrt(pow(fabs(r1), 2.0) + pow(fabs(r2), 2.0) + pow(fabs(r3), 2.0)) +
                     m1 * r1 * fabs(r1) * ((r1 / fabs(r1))) * 1.0 /
                         pow(pow(fabs(r1), 2.0) + pow(fabs(r2), 2.0) + pow(fabs(r3), 2.0), 3.0 / 2.0) +
                     m2 * r2 * fabs(r1) * ((r1 / fabs(r1))) * 1.0 /
                         pow(pow(fabs(r1), 2.0) + pow(fabs(r2), 2.0) + pow(fabs(r3), 2.0), 3.0 / 2.0) +
                     m3 * r3 * fabs(r1) * ((r1 / fabs(r1))) * 1.0 /
                         pow(pow(fabs(r1), 2.0) + pow(fabs(r2), 2.0) + pow(fabs(r3), 2.0), 3.0 / 2.0)) *
                    3.0 +
                r3 * fabs(r1) * ((r1 / fabs(r1))) *
                    (m1 * r1 * 1.0 / sqrt(pow(fabs(r1), 2.0) + pow(fabs(r2), 2.0) + pow(fabs(r3), 2.0)) +
                     m2 * r2 * 1.0 / sqrt(pow(fabs(r1), 2.0) + pow(fabs(r2), 2.0) + pow(fabs(r3), 2.0)) +
                     m3 * r3 * 1.0 / sqrt(pow(fabs(r1), 2.0) + pow(fabs(r2), 2.0) + pow(fabs(r3), 2.0))) *
                    1.0 / pow(pow(fabs(r1), 2.0) + pow(fabs(r2), 2.0) + pow(fabs(r3), 2.0), 3.0 / 2.0) * 3.0) *
                  1.0 / pow(pow(fabs(r1), 2.0) + pow(fabs(r2), 2.0) + pow(fabs(r3), 2.0), 3.0 / 2.0) +
              fabs(r1) * ((r1 / fabs(r1))) *
                  (m3 - r3 *
                            (m1 * r1 * 1.0 / sqrt(pow(fabs(r1), 2.0) + pow(fabs(r2), 2.0) + pow(fabs(r3), 2.0)) +
                             m2 * r2 * 1.0 / sqrt(pow(fabs(r1), 2.0) + pow(fabs(r2), 2.0) + pow(fabs(r3), 2.0)) +
                             m3 * r3 * 1.0 / sqrt(pow(fabs(r1), 2.0) + pow(fabs(r2), 2.0) + pow(fabs(r3), 2.0))) *
                            1.0 / sqrt(pow(fabs(r1), 2.0) + pow(fabs(r2), 2.0) + pow(fabs(r3), 2.0)) * 3.0) *
                  1.0 / pow(pow(fabs(r1), 2.0) + pow(fabs(r2), 2.0) + pow(fabs(r3), 2.0), 5.0 / 2.0) * 3.0;
    T[2][1] = -(r3 * 1.0 / sqrt(pow(fabs(r1), 2.0) + pow(fabs(r2), 2.0) + pow(fabs(r3), 2.0)) *
                    (-m2 * 1.0 / sqrt(pow(fabs(r1), 2.0) + pow(fabs(r2), 2.0) + pow(fabs(r3), 2.0)) +
                     m1 * r1 * fabs(r2) * ((r2 / fabs(r2))) * 1.0 /
                         pow(pow(fabs(r1), 2.0) + pow(fabs(r2), 2.0) + pow(fabs(r3), 2.0), 3.0 / 2.0) +
                     m2 * r2 * fabs(r2) * ((r2 / fabs(r2))) * 1.0 /
                         pow(pow(fabs(r1), 2.0) + pow(fabs(r2), 2.0) + pow(fabs(r3), 2.0), 3.0 / 2.0) +
                     m3 * r3 * fabs(r2) * ((r2 / fabs(r2))) * 1.0 /
                         pow(pow(fabs(r1), 2.0) + pow(fabs(r2), 2.0) + pow(fabs(r3), 2.0), 3.0 / 2.0)) *
                    3.0 +
                r3 * fabs(r2) * ((r2 / fabs(r2))) *
                    (m1 * r1 * 1.0 / sqrt(pow(fabs(r1), 2.0) + pow(fabs(r2), 2.0) + pow(fabs(r3), 2.0)) +
                     m2 * r2 * 1.0 / sqrt(pow(fabs(r1), 2.0) + pow(fabs(r2), 2.0) + pow(fabs(r3), 2.0)) +
                     m3 * r3 * 1.0 / sqrt(pow(fabs(r1), 2.0) + pow(fabs(r2), 2.0) + pow(fabs(r3), 2.0))) *
                    1.0 / pow(pow(fabs(r1), 2.0) + pow(fabs(r2), 2.0) + pow(fabs(r3), 2.0), 3.0 / 2.0) * 3.0) *
                  1.0 / pow(pow(fabs(r1), 2.0) + pow(fabs(r2), 2.0) + pow(fabs(r3), 2.0), 3.0 / 2.0) +
              fabs(r2) * ((r2 / fabs(r2))) *
                  (m3 - r3 *
                            (m1 * r1 * 1.0 / sqrt(pow(fabs(r1), 2.0) + pow(fabs(r2), 2.0) + pow(fabs(r3), 2.0)) +
                             m2 * r2 * 1.0 / sqrt(pow(fabs(r1), 2.0) + pow(fabs(r2), 2.0) + pow(fabs(r3), 2.0)) +
                             m3 * r3 * 1.0 / sqrt(pow(fabs(r1), 2.0) + pow(fabs(r2), 2.0) + pow(fabs(r3), 2.0))) *
                            1.0 / sqrt(pow(fabs(r1), 2.0) + pow(fabs(r2), 2.0) + pow(fabs(r3), 2.0)) * 3.0) *
                  1.0 / pow(pow(fabs(r1), 2.0) + pow(fabs(r2), 2.0) + pow(fabs(r3), 2.0), 5.0 / 2.0) * 3.0;
    T[2][2] = -((m1 * r1 * 1.0 / sqrt(pow(fabs(r1), 2.0) + pow(fabs(r2), 2.0) + pow(fabs(r3), 2.0)) +
                 m2 * r2 * 1.0 / sqrt(pow(fabs(r1), 2.0) + pow(fabs(r2), 2.0) + pow(fabs(r3), 2.0)) +
                 m3 * r3 * 1.0 / sqrt(pow(fabs(r1), 2.0) + pow(fabs(r2), 2.0) + pow(fabs(r3), 2.0))) *
                    1.0 / sqrt(pow(fabs(r1), 2.0) + pow(fabs(r2), 2.0) + pow(fabs(r3), 2.0)) * -3.0 +
                r3 * 1.0 / sqrt(pow(fabs(r1), 2.0) + pow(fabs(r2), 2.0) + pow(fabs(r3), 2.0)) *
                    (-m3 * 1.0 / sqrt(pow(fabs(r1), 2.0) + pow(fabs(r2), 2.0) + pow(fabs(r3), 2.0)) +
                     m1 * r1 * fabs(r3) * ((r3 / fabs(r3))) * 1.0 /
                         pow(pow(fabs(r1), 2.0) + pow(fabs(r2), 2.0) + pow(fabs(r3), 2.0), 3.0 / 2.0) +
                     m2 * r2 * fabs(r3) * ((r3 / fabs(r3))) * 1.0 /
                         pow(pow(fabs(r1), 2.0) + pow(fabs(r2), 2.0) + pow(fabs(r3), 2.0), 3.0 / 2.0) +
                     m3 * r3 * fabs(r3) * ((r3 / fabs(r3))) * 1.0 /
                         pow(pow(fabs(r1), 2.0) + pow(fabs(r2), 2.0) + pow(fabs(r3), 2.0), 3.0 / 2.0)) *
                    3.0 +
                r3 * fabs(r3) * ((r3 / fabs(r3))) *
                    (m1 * r1 * 1.0 / sqrt(pow(fabs(r1), 2.0) + pow(fabs(r2), 2.0) + pow(fabs(r3), 2.0)) +
                     m2 * r2 * 1.0 / sqrt(pow(fabs(r1), 2.0) + pow(fabs(r2), 2.0) + pow(fabs(r3), 2.0)) +
                     m3 * r3 * 1.0 / sqrt(pow(fabs(r1), 2.0) + pow(fabs(r2), 2.0) + pow(fabs(r3), 2.0))) *
                    1.0 / pow(pow(fabs(r1), 2.0) + pow(fabs(r2), 2.0) + pow(fabs(r3), 2.0), 3.0 / 2.0) * 3.0) *
                  1.0 / pow(pow(fabs(r1), 2.0) + pow(fabs(r2), 2.0) + pow(fabs(r3), 2.0), 3.0 / 2.0) +
              fabs(r3) * ((r3 / fabs(r3))) *
                  (m3 - r3 *
                            (m1 * r1 * 1.0 / sqrt(pow(fabs(r1), 2.0) + pow(fabs(r2), 2.0) + pow(fabs(r3), 2.0)) +
                             m2 * r2 * 1.0 / sqrt(pow(fabs(r1), 2.0) + pow(fabs(r2), 2.0) + pow(fabs(r3), 2.0)) +
                             m3 * r3 * 1.0 / sqrt(pow(fabs(r1), 2.0) + pow(fabs(r2), 2.0) + pow(fabs(r3), 2.0))) *
                            1.0 / sqrt(pow(fabs(r1), 2.0) + pow(fabs(r2), 2.0) + pow(fabs(r3), 2.0)) * 3.0) *
                  1.0 / pow(pow(fabs(r1), 2.0) + pow(fabs(r2), 2.0) + pow(fabs(r3), 2.0), 5.0 / 2.0) * 3.0;
    T *= 1.0 / (4 * pi);
    return T;
}
dvec3 Simulation::Hext(dvec3 r) {

    auto r_hat = normalize(r);
    auto H = 1 / (4 * pi) * ((3.0 * r_hat * dot(m, r_hat) - m) / (std::pow(length(r), 3)));
    return H;
}
void Simulation::eval_Hext() {
    // still need some thinking here
    // single point magnetic field
    // lets try with (0, 1, 0)
    // for (size_t i = 0; i < num_particles; i++) {
    //     pointers.Hext[i] = vec3(1, 0, 0);
    // }
    // const double mu0 = 1.25663706212e-16;

    for (size_t i = 0; i < num_particles; i++) {
        dvec3 p = pointers.particle_position[i];
        dvec3 r = p - dipole;
        pointers.Hext[i] = Hext(r);
    }
}

void Simulation::visualize_field(Eigen::MatrixXd &P, Eigen::MatrixXi &F) {
    P.resize(2000, 3);
    F.resize(1000, 2);
    for (int x = 0; x < 10; x++) {
        for (int y = 0; y < 10; y++) {
            for (int z = 0; z < 10; z++) {
                int idx = x + y * 10 + z * 100;
                Eigen::Vector3d p(x, y, z);
                p /= 10.0;
                P.row(idx) = p;
                auto h = Hext(dvec3(p[0], p[1], p[2]) - dipole);
                P.row(idx + 1000) = p + Eigen::Vector3d(h[0], h[1], h[2]).normalized() * 0.04;
                F.row(idx) = Eigen::RowVector2i(idx, idx + 1000);
            }
        }
    }
}

void Simulation::get_R(Eigen::Matrix3d &R, const Eigen::Vector3d &rt, const Eigen::Vector3d &rs) {
    // given rt and rs world coordinate, transform it to the coordinate where rs is on origin
    // let (xi, eta, zeta) be the unit vectors and assume rt is on its zeta axis.
    Eigen::Vector3d zeta = (rt - rs).normalized();
    Eigen::Vector3d z;
    if (abs(zeta.x()) < 1e-2) {
        z << 0, 0, 1;
    } else {
        z << 1, 0, 0;
    }
    Eigen::Vector3d eta = zeta.cross(z);
    eta.normalize();
    Eigen::Vector3d xi = eta.cross(zeta);
    R.col(0) = xi;
    R.col(1) = eta;
    R.col(2) = zeta;
}

// data provided in appendix
float Simulation::get_C1(float q) {
    float C1 = 0.0;
    if (0 < q && q <= 1) {
        C1 = q * (q * (q * (q * (9.97813616438174e-09) + (-2.97897856524718e-08)) + (2.38918644566813e-09)) +
                  (4.53199938857366e-08)) +
             (2.44617454752747e-11);
    } else if (1 < q && q <= 2) {
        C1 = q * (q * (q * (q * (-2.76473728643294e-09) + (2.86975546540539e-08)) + (-9.94582836806651e-08)) +
                  (1.25129924573675e-07)) +
             (-2.37010166723652e-08);
    } else if (2 < q && q <= 3) {
        C1 = q * (q * (q * (q * (-1.09679990621465e-09) + (9.77055663264614e-09)) + (-2.54781238661150e-08)) +
                  (2.65020634884934e-09)) +
             (5.00787562417835e-08);
    } else if (3 < q && q <= 4) {
        C1 = q * (q * (q * (q * (3.79927162333632e-10) + (-6.26368404962679e-09)) + (3.94760528277489e-08)) +
                  (-1.13580541622200e-07)) +
             (1.27491333574323e-07);
    }
    return C1;
}

float Simulation::get_C2(float q) {
    float C2 = 0.0;
    if (0 < q && q <= 1) {
        C2 = q * (q * (q * (q * (6.69550479838731e-08) + (-1.61753307173877e-07)) + (1.68213714992711e-08)) +
                  (1.34558143036838e-07)) +
             (1.10976027980100e-10);
    } else if (1 < q && q <= 2) {
        C2 = q * (q * (q * (q * (-3.08460139955194e-08) + (2.29192245602275e-07)) + (-5.88399621128587e-07)) +
                  (5.61170054591844e-07)) +
             (-1.14421132829680e-07);
    } else if (2 < q && q <= 3) {
        C2 = q * (q * (q * (q * (3.50477408060213e-09) + (-5.25956271895141e-08)) + (2.78876509535747e-07)) +
                  (-6.24199554212217e-07)) +
             (4.91807818904985e-07);
    } else if (3 < q && q <= 4) {
        C2 = q * (q * (q * (q * (7.33485346367840e-10) + (-9.58855788627803e-09)) + (4.37085309763591e-08)) +
                  (-7.48004594092261e-08)) +
             (2.34161209605651e-08);
    }
    return C2;
}

void Simulation::get_T_hat(Eigen::Matrix3d &T_hat, const Eigen::Vector3d &m_hat_s, float q) {
    float C1 = get_C1(q);
    float C2 = get_C2(q);
    T_hat << m_hat_s[2] * C1, 0, m_hat_s[0] * C1, 0, m_hat_s[2] * C1, m_hat_s[1] * C1, m_hat_s[0] * C1, m_hat_s[1] * C1,
        m_hat_s[2] * C2;
    T_hat /= h * h * h * h;
}

void Simulation::get_Force_Tensor(Eigen::Matrix3d &Ts, const Eigen::Vector3d &rt, const Eigen::Vector3d &rs,
                                  const Eigen::Vector3d &ms) {
    Eigen::Vector3d r = rt - rs;
    vec3 r_for_W = vec3(r[0], r[1], r[2]);
    float r_norm = length(r_for_W);
    // CHECK(W(r_for_W) == 0.0);
    // CHECK(r_norm >= 4.0 * h);
    auto W_avr_r = W_avr(r_for_W);
    auto W_r = 0.0; // W(r_for_W);
    float Ar = (W_avr_r - W_r) / (r_norm * r_norm);
    float Ar_prime = (5 * W_r) / (r_norm * r_norm * r_norm) - (5 * W_avr_r) / (r_norm * r_norm * r_norm) -
                     dWdr(r_for_W) / (r_norm * r_norm);
    Ts = (Eigen::Matrix3d::Identity() * (r.transpose() * ms) + r * ms.transpose() + ms * r.transpose()) * Ar +
         r * (r.transpose() * ms) * (r.transpose() / r_norm) * Ar_prime;
    // Ts *= mu0;
}

void my_far_force_tensor(glm::mat3 &Bij, const glm::vec3 &r_vec, const glm::vec3 &s_vec, const float &q,
                         const float &h) {
    //
    // Bij = I dot(r, m) A + rm' A + rr' dAdr - mr' dBdr
}

void Simulation::compute_m(const Eigen::VectorXd &b) {
    Eigen::VectorXd Gamma_b = Gamma * b;
    for (size_t i = 0; i < num_particles; i++) {
        pointers.particle_mag_moment[i] =
            vec3(Gamma_b.segment<3>(i * 3)[0], Gamma_b.segment<3>(i * 3)[1], Gamma_b.segment<3>(i * 3)[2]);
    }
}
void Simulation::magnetization() {
    tbb::parallel_for<size_t>(0, num_particles, [&](size_t t) {
        pointers.particle_H[t] = vec3(0.0f, 0.0f, 0.0f);
        pointers.particle_M[t] = vec3(0.0f, 0.0f, 0.0f);
        for (size_t s = 0; s < num_particles; s++) {
            if (t == s)
                continue;
            pointers.particle_H[t] +=
                H(pointers.particle_position[t] - pointers.particle_position[s], pointers.particle_mag_moment[s]);
            pointers.particle_M[t] +=
                pointers.particle_mag_moment[s] * W(pointers.particle_position[t] - pointers.particle_position[s]);
        }
    });
}

void Simulation::compute_magenetic_force() {
    eval_Hext();
    Eigen::VectorXd hext(3 * num_particles), b;
    for (size_t i = 0; i < num_particles; i++) {
        hext.segment<3>(3 * i) << pointers.Hext[i][0], pointers.Hext[i][1], pointers.Hext[i][2];
    }
    if (enable_interparticle_magnetization) {
        // magnetization();
        Eigen::SparseMatrix<double> G;
        G.resize(3 * num_particles, 3 * num_particles);
        Eigen::MatrixXd G_tmp;
        G_tmp.resize(3 * num_particles, 3);
        G_tmp.setZero();
        // G.setZero();
        std::vector<Eigen::Triplet<double>> trip;
#if 0
    for (size_t i = 0; i < num_particles; i++) {
        vec3 ri = pointers.particle_position[i];
        for (size_t j = 0; j < num_particles; j++) {
            vec3 rj = pointers.particle_position[j];
            // rj -= vec3(dipole);
            vec3 r = ri - rj;
            vec3 mj = pointers.particle_mag_moment[j];
            //   trip.push_back(Eigen::Triplet<double>(3 * j, 3 * j,
            // pointers.particle_position[j][0])); trip.push_back(Eigen::Triplet<double>(3 * j + 1, 3 * j + 1,
            // pointers.particle_position[j][1])); trip.push_back(Eigen::Triplet<double>(3 * j + 2, 3 * j + 2,
            // pointers.particle_position[j][2])); for (int k = 0; k < 3; k++) {
            //     trip.emplace_back(3 * k, 3 * k, r[k]);
            // }
            // for (int k = 0; k < 3; k++) {
            //     trip.emplace_back(3 * j + k, 3 * j + k, pointers.particle_H[j][k] + pointers.particle_M[j][k]);
            // }
            Eigen::Matrix3d Hi = H_mat(r, mj);
            Eigen::Matrix3d Wi = Eigen::Matrix3d::Identity() * W(r);
            // std::cout << Hi << std::endl;
            for (int a = 0; a < 3; a++) {
                for (int b = 0; b < 3; b++) {
                    trip.emplace_back(3 * j + a, 3 * j + b, Hi(a, b) + Wi(a, b));
                }
            }
        }
    }
#else
        tbb::parallel_for<size_t>(0, num_particles, [&](size_t j) {
            vec3 rj = pointers.particle_position[j];
            vec3 mj = pointers.particle_mag_moment[j];
            Eigen::Matrix3d tmp;
            tmp.setZero();
            for (size_t i = 0; i < num_particles; i++) {
                vec3 ri = pointers.particle_position[i];
                vec3 r = ri - rj;
                Eigen::Matrix3d Hi = H_mat(r, mj);
                Eigen::Matrix3d Wi = Eigen::Matrix3d::Identity() * W(r);
                tmp += Hi + Wi;
            }
            G_tmp.block<3, 3>(3 * j, 0) = tmp;
        });
        for (size_t j = 0; j < num_particles; j++) {
            for (int a = 0; a < 3; a++) {
                for (int b = 0; b < 3; b++) {
                    trip.emplace_back(3 * j + a, 3 * j + b, G_tmp(3 * j + a, b));
                }
            }
        }
#endif
        G.setFromTriplets(trip.begin(), trip.end());
        Eigen::SparseMatrix<double> ident;
        ident.resize(3 * num_particles, 3 * num_particles);
        ident.setIdentity();
        Eigen::SparseMatrix<double> A = G * Gamma - ident;
        Eigen::LeastSquaresConjugateGradient<Eigen::SparseMatrix<double>> cg;
        cg.compute(A);

        b = cg.solve(-1.0 * hext);
        printf("%lf\n", b.norm());
        compute_m(b);
        // b = cg.solve(-1.0 * hext);
    } else {
        b = hext;
        compute_m(b);
    }
    // std::cout << b << std::endl;
    // for (size_t t = 0; t < num_particles; t++) {
    tbb::parallel_for<size_t>(0, num_particles, [&](size_t t) {
#if 1
        Eigen::Vector3d m_hat, ft;
        Eigen::Matrix3d R, Ts, T_hat;

        float q;
        double dist;
        Eigen::Matrix3d U;
        U.setZero();
        Eigen::Vector3d rt, mt;
        rt << pointers.particle_position[t][0], pointers.particle_position[t][1], pointers.particle_position[t][2];
        mt << pointers.particle_mag_moment[t][0], pointers.particle_mag_moment[t][1],
            pointers.particle_mag_moment[t][2];
        for (size_t s = 0; s < num_particles; s++) {
            Ts.setZero();
            Eigen::Vector3d rs, ms;
            rs << pointers.particle_position[s][0], pointers.particle_position[s][1], pointers.particle_position[s][2];
            ms << pointers.particle_mag_moment[s][0], pointers.particle_mag_moment[s][1],
                pointers.particle_mag_moment[s][2];
            dist = (rt - rs).norm();
            if (dist < 4.0 * h) {
                q = dist / h;
                get_R(R, rt, rs); // note rs is the source!
                m_hat = R.transpose() * ms;
                get_T_hat(T_hat, m_hat, q);
                Ts = R * T_hat * R.transpose(); // * 10000000.0;
                // std::cout << Ts << std::endl;
            } else {
                get_Force_Tensor(Ts, rt, rs, ms);
                // Ts *= 0.1;
                Ts *= mu0;
            }
            U += Ts;
        }
        ft = U * mt;
        dvec3 F = dvec3(ft[0], ft[1], ft[2]);
        F += glm::dmat3(dHext(dvec3(pointers.particle_position[t]) - dipole)) * dvec3(mt[0], mt[1], mt[2]) * mu0;
        pointers.particle_mag_force[t] = vec3(F);
#else
        mat3 U(0.0);
        auto rt = pointers.particle_position[t];
        auto mt = pointers.particle_mag_moment[t];
        for (size_t s = 0; s < num_particles; s++) {
            if (s == t)
                continue;
            auto rs = pointers.particle_position[s];
            auto ms = pointers.particle_mag_moment[s];
            float r = glm::length(rt - rs);
            float q = r * (1 / h);
            vec3 r_vec = rt - rs;
            vec3 s_vec = ms;
            mat3 Bij(0.0);
            if (q > 4) {
                get_far_field_force_tensor(Bij, r_vec, s_vec, q, h);
            } else {
                get_near_field_force_tensor(Bij, r_vec, s_vec, q, h);
            }
            U += Bij;
        }
        auto ft = U * mt;
        ft += glm::dmat3(dHext(dvec3(pointers.particle_position[t]) - dipole)) * dvec3(mt[0], mt[1], mt[2]) * mu0;
        // ft += mu0 * mt 
        pointers.particle_mag_force[t] = ft;
// printf("%f\n", length(ft));
#endif
    });
}

void Simulation::run_step_adami() {
    if (n_iter == 0) {
        tbb::parallel_for(size_t(0), num_particles, [=](size_t id) { pointers.particle_mag_force[id] = vec3(0); });
        compute_magenetic_force();
    }
    build_grid();
    find_neighbors();
    tbb::parallel_for(size_t(0), num_particles, [=](size_t id) {
        vec3 dvdt = dvdt_full(id);
        vec3 f = pointers.particle_mag_force[id];
        if (!glm::any(glm::isnan(f)))
            pointers.dvdt[id] = dvdt + f / mass; // v(t + dt/2)
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
    });
    if (n_iter % 10 == 0) {
        printf("compute magnetic force; iter=%zu\n", n_iter);
        compute_magenetic_force();
    }
    tbb::parallel_for(size_t(0), num_particles, [=](size_t id) {
        vec3 dvdt = dvdt_full(id);
        vec3 f = pointers.particle_mag_force[id];
        // CHECK(!glm::any(glm::isnan(f)));
        if (!glm::any(glm::isnan(f)))
            pointers.dvdt[id] = dvdt + f / mass; // Insert magnetic force here
    });
    tbb::parallel_for(size_t(0), num_particles, [=](size_t id) {
        pointers.particle_velocity[id] += dt * 0.5f * pointers.dvdt[id];
        pointers.P[id] = P(id);
    });
    naive_collison_handling();
    // printf("step done\n");
}
void Simulation::run_step() {
    run_step_adami();
    n_iter++;
}
