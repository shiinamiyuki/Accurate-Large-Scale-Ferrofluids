#include "reconstruction.h"
#include "simulation.h"
#include <Eigen/Core>
#include <atomic>
#include <chrono>
#include <igl/opengl/glfw/Viewer.h>
#include <igl/writeOBJ.h>
#include <iostream>
#include <random>
#include <sstream>
double reconstruction_iso = 0.5;
Eigen::Vector3i reconstruction_res(200, 200, 200);
Simulation setup_ferro_success() {
    std::vector<vec3> particles;
    {
        std::random_device rd;
        std::uniform_real_distribution<float> dist;
        for (float x = 0.3; x < 0.7; x += 0.02) {
            for (float z = 0.3; z < 0.7; z += 0.02) {
                for (float y = 0.0; y < 0.1; y += 0.01) {
                    particles.emplace_back(x, y, z);
                }
            }
        }
    }
    Simulation sim(particles);

    sim.enable_ferro = true;
    sim.enable_gravity = false;
    sim.enable_interparticle_magnetization = false;
    sim.enable_interparticle_force = true;
    sim.dt = 0.0005;
    {
        sim.lower.x = 0.3;
        sim.lower.z = 0.3;
        sim.upper.x = 0.7;
        sim.upper.z = 0.7;
    }
    reconstruction_iso = 0.5;
    reconstruction_res = Eigen::Vector3i(150, 80, 150);
    return sim;
}
Simulation setup_ferro_with_gravity_success() { // gravity makes it more stable but makes spikes shorter
    std::vector<vec3> particles;
    {
        std::random_device rd;
        std::uniform_real_distribution<float> dist;
        for (float x = 0.3; x < 0.7; x += 0.02) {
            for (float z = 0.3; z < 0.7; z += 0.02) {
                for (float y = 0.0; y < 0.1; y += 0.01) {
                    particles.emplace_back(x, y, z);
                }
            }
        }
    }
    Simulation sim(particles);

    sim.enable_ferro = true;
    sim.enable_gravity = true;
    sim.enable_interparticle_magnetization = false;
    sim.enable_interparticle_force = true;
    // sim.alpha = 10.0;
    sim.dt = 0.0004;
    {
        sim.lower.x = 0.3;
        sim.lower.z = 0.3;
        sim.upper.x = 0.7;
        sim.upper.z = 0.7;
    }
    reconstruction_iso = 3.2;
    reconstruction_res = Eigen::Vector3i(150, 80, 150);
    return sim;
}
Simulation setup_ferro_no_interparticle() {
    std::vector<vec3> particles;
    {
        std::random_device rd;
        std::uniform_real_distribution<float> dist;
        for (float x = 0.3; x < 0.7; x += 0.02) {
            for (float z = 0.3; z < 0.7; z += 0.02) {
                for (float y = 0.0; y < 0.1; y += 0.01) {
                    particles.emplace_back(x, y, z);
                }
            }
        }
    }
    Simulation sim(particles);
    sim.alpha = 4.0; // sim.alpha = 1 makes the fluid really funny
    sim.enable_ferro = true;
    sim.enable_gravity = false;
    sim.enable_interparticle_magnetization = false;
    sim.enable_interparticle_force = false;
    sim.dt = 0.0001;
    {
        sim.lower.x = 0.3;
        sim.lower.z = 0.3;
        sim.upper.x = 0.7;
        sim.upper.z = 0.7;
    }
    reconstruction_iso = 0.5;
    reconstruction_res = Eigen::Vector3i(150, 80, 150);
    return sim;
}
Simulation setup_sph_fluid_crown() {
    std::vector<vec3> particles;
    {
        std::random_device rd;
        std::uniform_real_distribution<float> dist;
        for (float x = 0.0; x < 0.99; x += 0.02) {
            for (float z = 0.0; z < 0.99; z += 0.02) {
                for (float y = 0.0; y < 0.15; y += 0.02) {
                    particles.emplace_back(x, y, z);
                }
            }
        }
        for (float x = 0.4; x < 0.6; x += 0.02) {
            for (float z = 0.4; z < 0.6; z += 0.02) {
                for (float y = 0.7; y < 0.9; y += 0.02) {
                    vec3 c(0.5, 0.6, 0.5);
                    if (length(vec3(x, y, z) - vec3(c)) < 0.1)
                        particles.emplace_back(x, y, z);
                }
            }
        }
    }
    Simulation sim(particles);
    sim.enable_ferro = false;
    sim.enable_gravity = true;
    sim.enable_interparticle_magnetization = false;
    sim.enable_interparticle_force = false;
    sim.dt = 0.001;
    sim.alpha = 0.4;
    sim.tension = 1000.0;
    reconstruction_iso = 0.5;
    reconstruction_res = Eigen::Vector3i(150, 80, 150);
    return sim;
}
Simulation setup_sph_wave_impact() {
    std::vector<vec3> particles;
    std::vector<vec3> velocity;
    {
        std::random_device rd;
        std::uniform_real_distribution<float> dist;
        for (float x = 0.0; x < 0.99; x += 0.02) {
            for (float z = 0.0; z < 0.99; z += 0.02) {
                for (float y = 0.0; y < 0.15; y += 0.02) {
                    particles.emplace_back(x, y, z);
                    vec3 p = vec3(x, y, z);
                    vec3 v = (vec3(0.5, y, 0.5) - p);
                    velocity.emplace_back(v);
                }
            }
        }
    }
    Simulation sim(particles, velocity);
    sim.enable_ferro = false;
    sim.enable_gravity = true;
    sim.enable_interparticle_magnetization = false;
    sim.enable_interparticle_force = false;
    sim.dt = 0.0003;
    sim.alpha = 0.04;
    sim.tension = 100;
    sim.c0 = 30;
    reconstruction_iso = 0.3;
    reconstruction_res = Eigen::Vector3i(150, 150, 150);
    return sim;
}
bool write_obj_sequence = false;
int main(int argc, char **argv) {
    if (argc > 1) {
        if (std::strcmp(argv[1], "-s") == 0) {
            write_obj_sequence = true;
        }
    }

    std::atomic_bool run_sim = true;
    std::atomic_bool sim_ready = false;

    // auto sim = setup_ferro_success();
    // auto sim = setup_ferro_with_gravity_success();
    // auto sim = setup_sph_wave_impact();
    auto sim = setup_sph_fluid_crown();
    // auto sim = setup_ferro_no_interparticle();

    Eigen::MatrixXd PP;
    Eigen::MatrixXi PI;
    sim.visualize_field(PP, PI);
    bool flag = true;
    Eigen::MatrixXd P;
    P.resize(0, 3);
    std::mutex sim_lk;

    auto write_obj = [&] {
        Eigen::VectorXd mass, density;
        {
            std::lock_guard<std::mutex> lk(sim_lk);
            mass.resize(sim.num_particles);
            density.resize(sim.num_particles);
            mass.setConstant(sim.mass);
            for (size_t i = 0; i < sim.num_particles; i++) {
                density[i] = sim.pointers.density[i];
            }
        }
        Eigen::MatrixXd V;
        Eigen::MatrixXi F;
        reconstruct(V, F, P, reconstruction_res, mass, density, sim.h, reconstruction_iso);
        std::time_t result = std::time(nullptr);
        std::ostringstream os;
        os << "sim-" << result << ".obj";
        igl::writeOBJ(os.str(), V, F);
    };
    auto write_obj_seq = [&](size_t iter) {
        Eigen::VectorXd mass, density;
        {
            std::lock_guard<std::mutex> lk(sim_lk);
            mass.resize(sim.num_particles);
            density.resize(sim.num_particles);
            mass.setConstant(sim.mass);
            for (size_t i = 0; i < sim.num_particles; i++) {
                density[i] = sim.pointers.density[i];
            }
        }
        Eigen::MatrixXd V;
        Eigen::MatrixXi F;
        reconstruct(V, F, P, reconstruction_res, mass, density, sim.h, reconstruction_iso);
        printf("============== WRITE OBJ SEQUENCE =================\n");
        std::ostringstream os;
        std::time_t result = std::time(nullptr);
        os << "sim-" << result << "-iter-" << sim.n_iter << ".obj";
        igl::writeOBJ(os.str(), V, F);
    };
    P.resize(sim.buffers.num_particles, 3);
    std::thread sim_thd([&] {
        while (flag) {
            if (!run_sim) {
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
            } else {
                {
                    std::lock_guard<std::mutex> lk(sim_lk);
                    sim.run_step();
                    sim_ready = true;
                }
                if (write_obj_sequence && sim.n_iter % 100 == 0) {
                    write_obj_seq(sim.n_iter);
                }
            }
        }
    });

    using Viewer = igl::opengl::glfw::Viewer;
    igl::opengl::glfw::Viewer viewer;
    // viewer.data().set_edges(PP, PI, Eigen::RowVector3d(1, 0.47, 0.45));
    viewer.callback_post_draw = [&](Viewer &) -> bool {
        if (sim_ready) {
            sim_ready = false;
            for (size_t i = 0; i < sim.buffers.num_particles; i++) {
                auto p = sim.buffers.particle_position[i];
                P.row(i) = Eigen::RowVector3d(p.x, p.y, p.z);
            }
            viewer.data().point_size = 5;
            viewer.data().set_points(P, Eigen::RowVector3d(1, 1, 1));
        }
        return false;
    };
    viewer.callback_key_pressed = [&](Viewer &, unsigned int key, int) -> bool {
        if (key == ' ') {
            run_sim = !run_sim;
        } else if (key == 's') {
            write_obj();
        }
        return false;
    };
    viewer.core().is_animating = true;
    viewer.launch();
    flag = false;
    sim_thd.join();
}