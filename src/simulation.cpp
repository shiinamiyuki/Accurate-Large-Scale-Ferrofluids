#include "simulation.h"
#include <tbb/parallel_for.h>
#include <Eigen/Core>
#include <Eigen/Sparse>
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

vec3 Simulation::H(vec3 r, vec3 m){
    float r_norm = dot(r, r);
    vec3 r_hat = normalize(r);
    vec3 H_r = dot(r_hat, m) * (W_avr(r) - W(r)) * r_hat - (W_avr(r)/3.0f) * m;
    return H_r;
}

float Simulation::W_avr(vec3 r) { 
    float r_norm = dot(r, r);
    float W_r_h = 0.0;
    float q = r_norm/h;
    if (0 <= q && q < 1){
        W_r_h = (1.0 / 40.0) * (15.0 * pow(q, 3) - 36 * pow(q, 2) + 40.0);
    }
    else if (1 <= q && q < 2){
        W_r_h = (-3.0/(4.0 * pow(q, 3))) * (pow(q, 6)/6.0 - (6.0 * pow(q, 5))/5.0 + 
                3.0 * pow(q, 4) - (8.0 * pow(q, 3))/3.0 + 1.0/15.0);
    }
    else{
        W_r_h = 3.0 / (4.0 * pow(q, 3));
    }
    W_r_h *= (1.0 / pi);
    W_r_h *= (1.0 / pow(h, 3));
    return W_r_h;
}

float Simulation::W(vec3 r) {
    float r_norm = dot(r, r);
    float W_r_h = 0.0;
    float q = r_norm/h;
    if (0 <= q && q < 1){
        W_r_h = 0.25 * pow((2.0 - q), 3) - pow((1.0 - q), 3); 
    }
    if (1 <= q && q < 2){
        W_r_h = 0.25 * pow((2.0 - q), 3);
    }
    W_r_h *= (1.0 / pi);
    W_r_h *= (1.0 / pow(h, 3));
    return W_r_h;
}

float Simulation::dWdr(vec3 r) {
    float r_norm = dot(r, r);
    float dW_r_h = 0.0;
    float q = r_norm/h;
    if (0 <= q && q < 1){
        dW_r_h = 2.25 * pow(q, 2) - 3.0 * q;
    }
    if (1 <= q && q < 2){
        dW_r_h = -0.75 * pow(q, 2) + 3.0 * q - 3.0;
    }
    dW_r_h *= (1.0 / pi);
    dW_r_h *= (1.0 / pow(h, 3));
    return dW_r_h;
}

void Simulation::eval_Hext() {
    // still need some thinking here
    // single point magnetic field
}

void Simulation::get_R(Eigen::Matrix3d &R, const Eigen::Vector3d &rt, const Eigen::Vector3d &rs){
    // given rt and rs world coordinate, transform it to the coordinate where rs is on origin
    // let (xi, eta, zeta) be the unit vectors and assume rt is on its zeta axis.
    Eigen::Vector3d zeta = (rt - rs).normalized();
    Eigen::Vector3d y, z;
    y << 0, 1, 0;
    z << 0, 0, 1;
    Eigen::Vector3d eta = zeta.cross(z);
    eta[1] = 1e-10;
    eta.normalize();
    Eigen::Vector3d xi = eta.cross(zeta);
    R.col(0) = xi;
    R.col(1) = eta;
    R.col(2) = zeta;
}

// data provided in appendix
float Simulation::get_C1(float &q){
    float C1;
    if (0 < q&&q <= 1){
        C1 = q*(q*(q*(q*(9.97813616438174e-09)+(-2.97897856524718e-08))+(2.38918644566813e-09))+(4.53199938857366e-08))+(2.44617454752747e-11);
    }
    else if (1 < q&&q <= 2){
        C1 = q*(q*(q*(q*(-2.76473728643294e-09)+(2.86975546540539e-08))+(-9.94582836806651e-08))+(1.25129924573675e-07))+(-2.37010166723652e-08);
    }
    else if (2 < q&&q <= 3){
        C1 = q*(q*(q*(q*(-1.09679990621465e-09)+(9.77055663264614e-09))+(-2.54781238661150e-08))+(2.65020634884934e-09))+(5.00787562417835e-08);
    }
    else if (3 < q&&q <= 4){
        C1 = q*(q*(q*(q*(3.79927162333632e-10)+(-6.26368404962679e-09))+(3.94760528277489e-08))+(-1.13580541622200e-07))+(1.27491333574323e-07);
    }
    return C1;
}

float Simulation::get_C2(float &q){
    float C2;
    if (0 < q&&q <= 1){
        C2 = q*(q*(q*(q*(6.69550479838731e-08)+(-1.61753307173877e-07))+(1.68213714992711e-08))+(1.34558143036838e-07))+(1.10976027980100e-10);
    }
    else if (1 < q&&q <= 2){
        C2 = q*(q*(q*(q*(-3.08460139955194e-08)+(2.29192245602275e-07))+(-5.88399621128587e-07))+(5.61170054591844e-07))+(-1.14421132829680e-07);
    }
    else if (2 < q&&q <= 3){
        C2 = q*(q*(q*(q*(3.50477408060213e-09)+(-5.25956271895141e-08))+(2.78876509535747e-07))+(-6.24199554212217e-07))+(4.91807818904985e-07);
    }
    else if (3 < q&&q <= 4){
        C2 = q*(q*(q*(q*(7.33485346367840e-10)+(-9.58855788627803e-09))+(4.37085309763591e-08))+(-7.48004594092261e-08))+(2.34161209605651e-08);
    }
    return C2;
}

void Simulation::get_T_hat(Eigen::Matrix3d &T_hat, const Eigen::Vector3d &m_hat_s, float & q){
    float C1 = get_C1(q);
    float C2 = get_C2(q);
    T_hat << m_hat_s[2] * C1, 0, m_hat_s[0] * C1,
             0, m_hat_s[2] * C1, m_hat_s[1] * C1,
             m_hat_s[0] * C1, m_hat_s[1] * C1, m_hat_s[2] * C2;
}

void Simulation::get_Force_Tensor(Eigen::Matrix3d &Ts, const Eigen::Vector3d &rt, const Eigen::Vector3d &rs, const Eigen::Vector3d &ms){
    Eigen::Vector3d r = rt - rs;
    vec3 r_for_W = vec3(r[0], r[1], r[2]);
    float r_norm = dot(r_for_W, r_for_W);
    float Ar = (W_avr(r_for_W) - W(r_for_W))/pow(r_norm, 2);
    float Ar_prime = (5 * W(r_for_W))/pow(r_norm, 3) - (5 * W_avr(r_for_W))/pow(r_norm, 3) - dWdr(r_for_W)/pow(r_norm, 2);
    Ts = (Eigen::Matrix3d::Identity() * (r.transpose() * ms) + r * ms.transpose() + ms * r.transpose()) * Ar +
            r * (r.transpose() * ms) *(r.transpose()/r_norm) * Ar_prime;
}

void Simulation::compute_m(const Eigen::VectorXd &b){
    Eigen::VectorXd Gamma_b = Gamma * b;
    for(size_t i = 0; i < num_particles; i++){
        pointers.particle_mag_moment[i] = vec3(Gamma_b.segment<3>(i * 3)[0],Gamma_b.segment<3>(i * 3)[1], Gamma_b.segment<3>(i * 3)[2]);
    }
}
void Simulation::magnetization() {
    for(size_t t = 0; t < num_particles; t++){
        pointers.particle_H[t] = vec3(0.0f, 0.0f, 0.0f);
        pointers.particle_M[t] = vec3(0.0f, 0.0f, 0.0f);
        for(size_t s = 0; s < num_particles; s++){
            pointers.particle_H[t] += H(pointers.particle_position[t] - pointers.particle_position[s], pointers.particle_mag_moment[s]);
            pointers.particle_M[t] += pointers.particle_mag_moment[s] * W(pointers.particle_position[t] - pointers.particle_position[s]);
        }
    }
}

void Simulation::compute_magenetic_force() {
    Eigen::VectorXd hext(3 * num_particles), b;
    for(size_t i = 0; i < num_particles; i++){
        hext << pointers.Hext[i][0], pointers.Hext[i][1], pointers.Hext[i][2];
    }
    Eigen::SparseMatrix<double> G;
    G.resize(3 * num_particles, 3 * num_particles);
    std::vector<Eigen::Triplet<double>> trip;
    for(size_t j = 0; j < num_particles; j++){
        trip.push_back(Eigen::Triplet<double>(3 * j, 3 * j, pointers.particle_position[j][0]));
        trip.push_back(Eigen::Triplet<double>(3 * j + 1, 3 * j + 1, pointers.particle_position[j][1]));
        trip.push_back(Eigen::Triplet<double>(3 * j + 2, 3 * j + 2, pointers.particle_position[j][2]));
    }
    G.setFromTriplets(trip.begin(), trip.end());
    Eigen::SparseMatrix<double> ident;
    ident.setIdentity();
    Eigen::SparseMatrix<double> A = G * Gamma - ident;
    Eigen::ConjugateGradient<Eigen::SparseMatrix<double>>cg;
    cg.compute(A);
    b = cg.solve(-1.0 * hext);
    b = cg.solve(-1.0 * hext);
    compute_m(b);
    Eigen::Vector3d m_hat, ft;
    Eigen::Matrix3d R, Ts, T_hat;
    float q;
    double dist;
    for(size_t t = 0; t < num_particles; t++){
        Eigen::Matrix3d U;
        U.setZero();
        Eigen::Vector3d rt, mt;
        rt << pointers.particle_position[t][0], pointers.particle_position[t][1], pointers.particle_position[t][2];
        mt << pointers.particle_mag_moment[t][0], pointers.particle_mag_moment[t][1], pointers.particle_mag_moment[t][2];
        for(size_t s = 0; s < num_particles; s++){
            Eigen::Vector3d rs, ms;
            rs << pointers.particle_position[s][0], pointers.particle_position[s][1], pointers.particle_position[s][2];
            ms << pointers.particle_mag_moment[s][0], pointers.particle_mag_moment[s][1], pointers.particle_mag_moment[s][2];
            dist = (rt-rs).norm();
            if ( dist < 4.0 * h){
                q = dist/h;
                get_R(R, rt, rs); //note rs is the source!
                m_hat = R.transpose() * ms;
                get_T_hat(T_hat, m_hat, q);
                Ts = R * T_hat * R.transpose();
            }
            else{
                get_Force_Tensor(Ts, rt, rs, ms);
            }
            U += Ts;
        }
        ft = U * mt;
        pointers.particle_mag_force[t] = vec3(ft[0], ft[1], ft[2]);
    }
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
