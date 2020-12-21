#pragma once

#include <glm/glm.hpp>
using glm::dvec3;
using glm::ivec2;
using glm::ivec3;
using glm::mat3;
using glm::vec2;
using glm::vec3;

// Code from author
// Only for debugging purpose and are not used in our experimeents
// https://github.com/ureternalreward/magnetic-force/blob/master/magnetic_solver/magnetic_force.cu

#define DEF_POLY4_FUNC_naive(_FUNCNAME, ARG1, ARG2, ARG3, ARG4, ARG5)                                                  \
    template <typename _Ty> _Ty _FUNCNAME(_Ty x) {                                                                     \
        return x * (x * (x * (x * (ARG1) + (ARG2)) + (ARG3)) + (ARG4)) + (ARG5);                                       \
    }

DEF_POLY4_FUNC_naive(c1p0, 9.97813616438174e-09, -2.97897856524718e-08, 2.38918644566813e-09, 4.53199938857366e-08,
                     2.44617454752747e-11);
DEF_POLY4_FUNC_naive(c1p1, -2.76473728643294e-09, 2.86975546540539e-08, -9.94582836806651e-08, 1.25129924573675e-07,
                     -2.37010166723652e-08);
DEF_POLY4_FUNC_naive(c1p2, -1.09679990621465e-09, 9.77055663264614e-09, -2.54781238661150e-08, 2.65020634884934e-09,
                     5.00787562417835e-08);
DEF_POLY4_FUNC_naive(c1p3, 3.79927162333632e-10, -6.26368404962679e-09, 3.94760528277489e-08, -1.13580541622200e-07,
                     1.27491333574323e-07);

DEF_POLY4_FUNC_naive(c2p0, 6.69550479838731e-08, -1.61753307173877e-07, 1.68213714992711e-08, 1.34558143036838e-07,
                     1.10976027980100e-10);
DEF_POLY4_FUNC_naive(c2p1, -3.08460139955194e-08, 2.29192245602275e-07, -5.88399621128587e-07, 5.61170054591844e-07,
                     -1.14421132829680e-07);
DEF_POLY4_FUNC_naive(c2p2, 3.50477408060213e-09, -5.25956271895141e-08, 2.78876509535747e-07, -6.24199554212217e-07,
                     4.91807818904985e-07);
DEF_POLY4_FUNC_naive(c2p3, 7.33485346367840e-10, -9.58855788627803e-09, 4.37085309763591e-08, -7.48004594092261e-08,
                     2.34161209605651e-08);

void evaluate_curve1_2(float &C1, float &C2, const float &q) {
    if (0 < q && q <= 1) {
        C1 = c1p0(q);
        C2 = c2p0(q);
    } else if (1 < q && q <= 2) {
        C1 = c1p1(q);
        C2 = c2p1(q);
    } else if (2 < q && q <= 3) {
        C1 = c1p2(q);
        C2 = c2p2(q);
    } else if (3 < q && q <= 4) {
        C1 = c1p3(q);
        C2 = c2p3(q);
    }
}

void get_near_field_force_tensor(glm::mat3 &Bij, const glm::vec3 &r_vec, const glm::vec3 &s_vec, const float &q,
                                 const float &h) {
    // curve 1 and 2 are the two non zero component in Tijk tensor
    // Tijk(q,h): i force dir, J source dir, k target dir
    // gives the force when source and target are on z axis.
    // 6 component have curve1, Tzzz have curve 2.
    float curve1_q_hm4;
    float curve2_q_hm4;
    if (q == 0) {
        // when two particles overlap, no force
        return;
    }
    evaluate_curve1_2(curve1_q_hm4, curve2_q_hm4, q);
    curve1_q_hm4 *= 1 / (h * h * h * h);
    curve2_q_hm4 *= 1 / (h * h * h * h);
    // the non-zero component in the unrotated tensor is complete
    // time to calculate the rotated tensor.
    glm::vec3 zeta = glm::normalize(r_vec);
    glm::vec3 y{0, 1, 0};
    glm::vec3 z{0, 0, 1};
    glm::vec3 eta = glm::cross(zeta, z);
    // in case eta=0 because zeta is on z direction
    // make default the y direction.
    eta.y += 1e-10;
    eta = glm::normalize(eta);

    glm::vec3 xi = glm::cross(eta, zeta);

    // constructor using three column vectors
    glm::mat3 M{xi, eta, zeta};
    // note in GLM matrix, M[i] gives the ith column!

    // calculation formula:
    // Bal=M^T_(i a)T_(i j k)(q,h)M^T_(j p)(m_p)~M^T_(k l)
    // a is the force direction in world, l is the target dipole direction in world

    // source in the rotated axis:
    // M^float *s_vec
    glm::vec3 s_xez = s_vec * M;

    // U=T_(i j k)(q,h)M^T_(j p)(m_p)^~
    // is the center matrix
    // this constructor gives first column vector, then second
    glm::mat3 U{s_xez.z * curve1_q_hm4, 0,
                s_xez.x * curve1_q_hm4, 0,
                s_xez.z * curve1_q_hm4, s_xez.y * curve1_q_hm4,
                s_xez.x * curve1_q_hm4, s_xez.y * curve1_q_hm4,
                s_xez.z * curve2_q_hm4};

    ////U1,U2,U3 row vector
    // glm::vec3 U1{ s_xez.z*C1,0                   ,s_xez.x*C1 };
    // glm::vec3 U2{ 0,                   s_xez.z*C1,s_xez.y*C1 };
    // glm::vec3 U3{ s_xez.x*C1,s_xez.y*C1,s_xez.z*C2 };
    //
    ////U*M^float
    // U1 = U1*M;
    // U2 = U2*M;
    // U3 = U3*M;

    Bij += M * U * glm::transpose(M);
}

void get_far_field_force_tensor(glm::mat3 &Bij, const glm::vec3 &r_vec, const glm::vec3 &s_vec, const float &q,
                                const float &h) {
    constexpr double pi = 3.1415926535897;
    float r = q * h;
    float w_avr = 3.f / (4 * pi * r * r * r);
    float invr = 1;
    if (r != 0) {
        invr = float(1) / r;
    }

    // calculate the gradHferro at this point
    // formulas see plan.tm
    const float mu_0 = 1.25663706e-6;
    float A = mu_0 * invr * invr * (w_avr);
    float rdotm = dot(r_vec, s_vec);
    float dAdr = mu_0 * invr * invr * (5 * invr * (-w_avr)) * rdotm * invr;
    /*dBdr = 3 * invr*(w_dens - w_avr)*invr/3;*/
    float dBdr = mu_0 * invr * (-w_avr) * invr;

    const float &rx = r_vec.x;
    const float &ry = r_vec.y;
    const float &rz = r_vec.z;

    // Gxx
    Bij[0][0] += rx * s_vec.x * A + rx * dAdr * rx - s_vec.x * dBdr * rx;
    Bij[0][0] += rdotm * A;
    // Gxy
    // Gxy->B01->Bij[1][0]
    Bij[1][0] += rx * s_vec.y * A + rx * dAdr * ry - s_vec.x * dBdr * ry;
    // Gxz
    // Gxz->B02->Bij[2][0]
    Bij[2][0] += rx * s_vec.z * A + rx * dAdr * rz - s_vec.x * dBdr * rz;

    // Gyx
    Bij[0][1] += ry * s_vec.x * A + ry * dAdr * rx - s_vec.y * dBdr * rx;
    // Gyy
    Bij[1][1] += ry * s_vec.y * A + ry * dAdr * ry - s_vec.y * dBdr * ry;
    Bij[1][1] += rdotm * A;
    // Gyz
    Bij[2][1] += ry * s_vec.z * A + ry * dAdr * rz - s_vec.y * dBdr * rz;

    // Gzx
    Bij[0][2] += rz * s_vec.x * A + rz * dAdr * rx - s_vec.z * dBdr * rx;
    // Gzy
    Bij[1][2] += rz * s_vec.y * A + rz * dAdr * ry - s_vec.z * dBdr * ry;
    // Gzz
    Bij[2][2] += rz * s_vec.z * A + rz * dAdr * rz - s_vec.z * dBdr * rz;
    Bij[2][2] += rdotm * A;
}
