#include <iostream>
#include <armadillo>
#include <random>
#include <cmath>
#include <glpk.h>
#include "dikin_walk.h"

using namespace std;
using namespace arma;


void linprog(vec& c, mat& A_ub, mat& b_ub, mat& result) {
    //    Minimize: c^T * x
    //
    //    Subject to: A_ub * x <= b_ub
    //    A_eq * x == b_eq

    glp_prob* lp = glp_create_prob();
    glp_set_obj_dir(lp, GLP_MIN);
    glp_add_rows(lp, A_ub.n_rows);
    for (int i = 1; i <= A_ub.n_rows; ++i) {
        glp_set_row_bnds(lp, i, GLP_UP, 0.0, b_ub[i - 1]);
    }

    glp_add_cols(lp, A_ub.n_cols);
    for (int i = 1; i <= A_ub.n_cols; ++i) {
        glp_set_col_bnds(lp, i, GLP_FR, 0.0, 0.0);
        glp_set_obj_coef(lp, i, c[i - 1]);
    }

    int* ia = new int[A_ub.n_elem + 1];
    int* ja = new int[A_ub.n_elem + 1];
    double* ar = new double[A_ub.n_elem + 1];
    int i = 1;
    for (int r = 0; r < A_ub.n_rows; ++r) {
        for (int c = 0; c < A_ub.n_cols; ++c) {
            ar[i] = A_ub(r, c);
            ia[i] = r + 1;
            ja[i] = c + 1;
            i++;
        }
    }

    glp_load_matrix(lp, A_ub.n_elem, ia, ja, ar);

    glp_simplex(lp, NULL);

    // here we do not need the last element
    result.set_size(1, A_ub.n_cols - 1);
    for (int i = 0; i < A_ub.n_cols - 1; ++i) {
        result[i] = glp_get_col_prim(lp, i + 1);
    }

    glp_delete_prob(lp);
    delete [] ia;
    delete [] ja;
    delete [] ar;
}

void chebyshev_center(mat& a, mat& b, mat& x0) {
    // Return Chebyshev center of the convex polytope.
    mat norm_vector;
    norm_vector.set_size(a.n_rows, 1);
    for (int a_row = 0; a_row < a.n_rows; ++a_row) {

        double sum = 0;
        for (auto it = a.begin_row(a_row), e = a.end_row(a_row); it != e; ++it) {
            double elm = *it;
            sum += (elm * elm);
        }
        norm_vector(a_row, 0) = std::sqrt(sum);
    }
    norm_vector.print("norm_vec = ");

    vec c;
    c.zeros(a.n_cols + 1);
    c[a.n_cols] = -1;

    mat a_lp = join_rows(a, norm_vector);
    a_lp.print("a_lp = ");
    linprog(c, a_lp, b, x0);
}

mat hessian(mat& a, mat& b, mat& x) {
    //Return log-barrier Hessian matrix at x
    mat d = b - trans(a * trans(x));
    mat s = arma::pow(d, -2);
    mat ret = trans(a) * (diagmat(s)) * a;
    return ret;
}

double local_norm(mat& h, mat& v) {
    // Return the local norm of v based on the given Hessian matrix.
    mat ln = (v * h * trans(v));
    return ln[0];
}

mat sample_ellipsoid(mat& e, double r) {
    // Return a point in the (hyper)ellipsoid uniformly sampled.
    //
    // The ellipsoid is defined by the positive definite matrix, ``e``, and
    // the radius, ``r``.

    // Generate a point on the sphere surface
    std::random_device rd{};
    std::mt19937 gen{rd()};
    std::normal_distribution<> d{0, 1};
    mat p;
    p.set_size(1, e.n_rows);
    double norm = 0;
    for (int i = 0; i < e.n_rows; ++i) {
        double rd = d(gen);
        p[i] = rd;
        norm += (rd * rd);
    }
    p = p / std::sqrt(norm);

    // Scale to a point in the sphere volume
    std::uniform_real_distribution<> dis(0, 1);
    double rdnum = std::pow(dis(gen), 1.0/e.n_rows);
    p = p * rdnum;

    return trans(std::sqrt(r) * chol(inv(e)) * trans(p));
}

mat dikin_walk(mat& a, mat& b, mat& x, mat& h_x, double r) {
    umat less_or_eq = (trans(a * trans(x)) <= b);
    if (!arma::all(vectorise(less_or_eq))) {
        (trans(a * trans(x)) - b).print("Invalid state: ");
        exit(1);
    }

    std::random_device rd{};
    std::mt19937 gen{rd()};
    std::uniform_real_distribution<> dis(0, 1);
    if (dis(gen) < 0.5) {
        return x;
    }

    mat z = x + sample_ellipsoid(h_x, r);
    mat h_z = hessian(a, b, z);
    mat xminusz = x - z;

    if (local_norm(h_z, xminusz) > 1.0) {
        return x;
    }

    double p = std::sqrt(det(h_z) / det(h_x));
    if (p >= 1 || dis(gen) < p) {
        x = z;
        // h_x = h_z
        int i = 0;
        for (auto it = h_z.begin(), e = h_z.end(); it != e; ++it) {
            h_x[i++] = *it;
        }
    }
    return x;
}

void dikin_walk(mat& a, mat& b, mat& x0, mat& points, int dikin_radius, int burn, int thin, int count) {
    mat h_x = hessian(a, b, x0);
    h_x.print("h_x = ");
    points.set_size(count, x0.n_cols);

    int counted = 0;
    int burned = 0;
    int thined = thin;
    while (counted < count) {
        // produce point
        mat point = dikin_walk(a, b, x0, h_x, dikin_radius);

        if (burned < burn) {
            // throw away the first ``burn'' points
            burned++;
            continue;
        } else if (thined == thin) {
            // collect
            for (int i = 0; i < point.n_elem; ++i) {
                points(counted, i) = point[i];
            }
            point.print("points[i] = ");

            counted++;
            thined = 0;
        } else {
            // skip ``thin'' points before collect one
            thined++;
            continue;
        }
    }
}

void dikin_walk(mat& eq, mat& eq_rhs, mat& leq, mat& leq_rhs, mat& points, int count) {
    // Find nullspace
    mat U, V;
    vec s;
    bool ret = svd(U,s,V,eq);
    if (!ret) {
        cout << "fail to svd\n";
        exit(1);
    }
    s.print("s= ");
    int r = arma::rank(s);
    mat nullspace;
    if (r == 0) {
        cout << "No equality constraints given...\n";
        nullspace.eye(V.n_rows, V.n_rows);
    } else if (r == V.n_rows) {
        cout << "Only one solution in null space\n";
        exit (1);
    } else {
        int nullity = V.n_rows - r;
        nullspace.set_size(nullity, V.n_cols);

        for (int row = V.n_rows - nullity; row < V.n_rows; ++row) {
            int col = 0;
            for (auto it = V.begin_row(row), e = V.end_row(row); it != e; ++it) {
                nullspace(row - V.n_rows + nullity, col++) = *it;
            }
        }
        inplace_trans(nullspace);
    }
    nullspace.print("nullspace= ");

    // Polytope parameters
    mat a = leq * nullspace;
    mat& b = leq_rhs;
    // Initial point to start the chains from.
    // Use the Chebyshev center.
    mat x0;
    chebyshev_center(a, b, x0);

    (x0 * trans(nullspace)).print("Chebyshev center: ");
    a.print("A= ");
    b.print("b= ");
    x0.print("x0= ");

    int chain_count = 1;
    int burn = 1000;
    int thin = 10;

    dikin_walk(a, b, x0, points, 1, burn, thin, count);
}

int main() {
    // This example is based on a system of linear equalities and
    // inequalities. The convex polytope to sample is the nullspace of the
    // given system.
    //
    // Equalities
    // 1) x3 == 0
    mat eq;
    mat eq_rhs;
    eq << 0 << 0 << 1;
    eq_rhs << 0;

    // Inequalities
    // 1) -3*x1 - 2*x2 <= -6
    // 2) -x2 <= -1
    // 3) x1 - x2 <= 8
    // 4) -3*x1 + x2 <= 4
    // 5) x1 + 3*x2 <= 22
    // 6) x1 <= 10
    mat leq;
    mat leq_rhs;
    leq << -3 << -2 << 0 << endr
        << 0  << -1 << 0 << endr
        << 1  << -1 << 0 << endr
        << -3 << 1  << 0 << endr
        << 1  << 3  << 0 << endr
        << 1  << 0  << 0;
    leq_rhs << -6 << -1 << 8 << 4 << 22 << 10;

    int count = 2; // how many points to generate
    mat points;
    dikin_walk(eq, eq_rhs, leq, leq_rhs, points, count);

    return 0;
}
