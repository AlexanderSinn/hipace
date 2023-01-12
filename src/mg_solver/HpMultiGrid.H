/* Copyright 2022
 *
 * This file is part of HiPACE++.
 *array_rhs
 * Authors: Weiqun Zhang
 * License: BSD-3-Clause-LBNL
 */
#ifndef HIPACE_MULTIGRID_H_
#define HIPACE_MULTIGRID_H_

#include "utils/HipaceProfilerWrapper.H"
#include <AMReX_FArrayBox.H>
#include <AMReX_Geometry.H>
#include <type_traits>

/** brief namespace for Hipace Multigrid */
namespace hpmg {

template <class T>
struct ArrayC {
    T* AMREX_RESTRICT p;
    amrex::Long nstride = 0;
    struct {
        int x = 0;
        int y = 0;
    } end;

    ArrayC (const amrex::Array4<T>& rhs) noexcept
        : p(rhs.p),
          nstride(rhs.nstride),
          end{rhs.end.x, rhs.end.y}
    {
        if (p) {
            AMREX_ALWAYS_ASSERT(rhs.begin.x == 0);
            AMREX_ALWAYS_ASSERT(rhs.begin.y == 0);
            AMREX_ALWAYS_ASSERT(rhs.begin.z == 0);
            AMREX_ALWAYS_ASSERT(rhs.end.z == 1);
        }
        // slice is only one cell thick if allocated, begins at zero
        AMREX_ALWAYS_ASSERT(!rhs.p || rhs.begin.x == 0 && rhs.begin.y == 0 &&
                            rhs.begin.z == 0 && rhs.end.z == 1);
    }

    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    T& operator() (int i, int j, int n) const noexcept {
#if defined(AMREX_DEBUG) || defined(AMREX_BOUND_CHECK)
        if (i < 0 || i <= end.x || j < 0 || j <= end.y) {
            amrex::Abort();
        }
#endif
        const int color = (i + j) & 1;
        return p[((end.x * (end.y * color + j) + i) >> 1) + n * nstride];
    }
};

/** \brief Multigrid solver
 *
 * This solves `-acoef * sol + Lap(sol) = rhs` with homogeneous Dirichlet BC
 * on a 2D slice.  It can solve two types of linear systems.
 *
 * (1) sol and rhs have two components, whereas acoef has only one
 *     component.  For Type I, call solve1(...).
 *
 * (2) acoef, sol, rhs are complex numbers. The system is equivalent to
 *       -acoef_real * sol_real + acoef_imag * sol_imag + Lap(sol_real) = rhs_real
 *       -acoef_imag * sol_real - acoef_real * sol_imag + Lap(sol_imag) = rhs_imag
 *     For Type II, call solve2(...).  Here, acoef_real and acoef_imag can be
 *     either a scalar constant or FArrayBox.
 */
class MultiGrid
{
public:

    /** \brief Ctor
     *
     * \param[in] dx Cell spacing in x direction
     * \param[in] dy Cell spacing in y direction
     * \param[in] a_domain Box describing a 2D slice
     */
    explicit MultiGrid (amrex::Real dx, amrex::Real dy, amrex::Box a_domain);

    /** \brief Dtor */
    ~MultiGrid ();

    /** \brief Solve the Type I equation given the initial guess, right hand side,
     * and the coefficient.
     *
     * \param[in,out] sol the initial guess and final solution
     * \param[in] rhs right hand side
     * \param[in] acoef the coefficient
     * \param[in] tol_rel relative tolerance
     * \param[in] tol_abs absolute tolerance
     * \param[in] nummaxiter maximum number of iterations
     * \param[in] verbose verbosity level
     */
    void solve1 (amrex::FArrayBox& sol, amrex::FArrayBox const& rhs, amrex::FArrayBox const& acoef,
                 amrex::Real const tol_rel, amrex::Real const tol_abs, int const nummaxiter,
                 int const verbose);

    /** \brief Solve the Type II equation given the initial guess, right hand side,
     * and the coefficient.
     *
     * \param[in,out] sol the initial guess and final solution
     * \param[in] rhs right hand side
     * \param[in] acoef_real the constant real part of the coefficient
     * \param[in] acoef_imag the constant imaginary part of the coefficient
     * \param[in] tol_rel relative tolerance
     * \param[in] tol_abs absolute tolerance
     * \param[in] nummaxiter maximum number of iterations
     * \param[in] verbose verbosity level
     */
    void solve2 (amrex::FArrayBox& sol, amrex::FArrayBox const& rhs,
                 amrex::Real const acoef_real, amrex::Real const acoef_imag,
                 amrex::Real const tol_rel, amrex::Real const tol_abs,
                 int const nummaxiter, int const verbose);

    /** \brief Solve the Type II equation given the initial guess, right hand side,
     * and the coefficient.
     *
     * \param[in,out] sol the initial guess and final solution
     * \param[in] rhs right hand side
     * \param[in] acoef_real the constant real part of the coefficient
     * \param[in] acoef_imag the array imaginary part of the coefficient
     * \param[in] tol_rel relative tolerance
     * \param[in] tol_abs absolute tolerance
     * \param[in] nummaxiter maximum number of iterations
     * \param[in] verbose verbosity level
     */
    void solve2 (amrex::FArrayBox& sol, amrex::FArrayBox const& rhs,
                amrex::Real const acoef_real, amrex::FArrayBox const& acoef_imag,
                amrex::Real const tol_rel, amrex::Real const tol_abs,
                int const nummaxiter, int const verbose);

    /** \brief Solve the Type II equation given the initial guess, right hand side,
     * and the coefficient.
     *
     * \param[in,out] sol the initial guess and final solution
     * \param[in] rhs right hand side
     * \param[in] acoef_real the array real part of the coefficient
     * \param[in] acoef_imag the constant imaginary part of the coefficient
     * \param[in] tol_rel relative tolerance
     * \param[in] tol_abs absolute tolerance
     * \param[in] nummaxiter maximum number of iterations
     * \param[in] verbose verbosity level
     */
    void solve2 (amrex::FArrayBox& sol, amrex::FArrayBox const& rhs,
                amrex::FArrayBox const& acoef_real, amrex::Real const acoef_imag,
                amrex::Real const tol_rel, amrex::Real const tol_abs,
                int const nummaxiter, int const verbose);

    /** \brief Solve the Type II equation given the initial guess, right hand side,
     * and the coefficient.
     *
     * \param[in,out] sol the initial guess and final solution
     * \param[in] rhs right hand side
     * \param[in] acoef_real the array real part of the coefficient
     * \param[in] acoef_imag the array imaginary part of the coefficient
     * \param[in] tol_rel relative tolerance
     * \param[in] tol_abs absolute tolerance
     * \param[in] nummaxiter maximum number of iterations
     * \param[in] verbose verbosity level
     */
    void solve2 (amrex::FArrayBox& sol, amrex::FArrayBox const& rhs,
                amrex::FArrayBox const& acoef_real, amrex::FArrayBox const& acoef_imag,
                amrex::Real const tol_rel, amrex::Real const tol_abs,
                int const nummaxiter, int const verbose);

    /** \brief Average down the coefficient.  Ideally, this function is not
     * supposed to be a public function.  It's made public due to a CUDA
     * limitation. */
    void average_down_acoef ();
    /** \brief Perform a V-cycle.  Ideally, this function is not supposed to
     * be a public function.  It's made public due to a CUDA limitation. */
    void vcycle ();
    /** \brief Solve at the bottom of the V-cycle.  Ideally, this function
     * is not supposed to be a public function.  It's made public due to a
     * CUDA limitation. */
    void bottomsolve ();
    /** \brief Private function used by solve1 and solve2.  It's made public
     * due to a CUDA limitation. */
    void solve_doit (amrex::FArrayBox& sol, amrex::FArrayBox const& rhs,
                     amrex::Real const tol_rel, amrex::Real const tol_abs,
                     int const nummaxiter, int const verbose);
    /** \brief Centers the input box in x and y around the domain so that only the ghost
     * cells "overhang". Make it a slab in the z direction and set the index to 0.
     */
    amrex::Box center_box (amrex::Box in_box, amrex::Box domain) {
        amrex::Box out_box = amrex::makeSlab(
            in_box + (domain.smallEnd() + domain.bigEnd() - in_box.smallEnd() - in_box.bigEnd())/2,
        2, 0);
        AMREX_ALWAYS_ASSERT(out_box.contains(domain));
        return out_box;
    }

private:

    static constexpr int m_num_system_types = 2;
    int m_system_type = 0;

    /** 2D slice domain */
    amrex::Vector<amrex::Box> m_domain;
    /** Cell sizes */
    amrex::Real m_dx, m_dy;

    /** Bottom MG level */
    int m_max_level;
    /** The level below which a single block kernel is used */
    int m_single_block_level_begin;
    /** Number of MG levels */
    int m_num_mg_levels;
    /** Number of single-block-kernel levels */
    int m_num_single_block_levels;

    /** Alias to the solution argument passed in solve() */
    amrex::FArrayBox m_sol;
    /** Alias to the RHS argument passed in solve() */
    amrex::FArrayBox m_rhs;

    /** Number of temporary fabs needed */
    static constexpr int nfabvs = 4;
    /** Fabs for coefficient, one for each level */
    amrex::Vector<amrex::FArrayBox> m_acf;
    /** Fabs for residual, one for each level */
    amrex::Vector<amrex::FArrayBox> m_res;
    /** Fabs for correction, one for each level */
    amrex::Vector<amrex::FArrayBox> m_cor;
    /** Fabs for residual of the residual-correction form, one for each level */
    amrex::Vector<amrex::FArrayBox> m_rescor;

    /** Device pointer to ArrayCs used by the single-block kernel at the bottom */
    ArrayC<amrex::Real> const* m_acf_a = nullptr;
    ArrayC<amrex::Real> const* m_res_a = nullptr;
    ArrayC<amrex::Real> const* m_cor_a = nullptr;
    ArrayC<amrex::Real> const* m_rescor_a = nullptr;

    /** Pinned vector as a staging area for memcpy to device */
    amrex::Gpu::PinnedVector<ArrayC<amrex::Real> > m_h_ArrayC;
    /** Device vector of ArrayCs used by the single-block kernel at the bottom */
    amrex::Gpu::DeviceVector<ArrayC<amrex::Real> > m_d_ArrayC;

#if defined(AMREX_USE_CUDA)
    /** CUDA graphs for average-down */
    bool m_cuda_graph_acf_created[m_num_system_types] = {false,false};
    cudaGraph_t m_cuda_graph_acf[m_num_system_types] = {NULL,NULL};
    cudaGraphExec_t m_cuda_graph_exe_acf[m_num_system_types] = {NULL,NULL};

    /** CUDA graphs for the V-cycle*/
    bool m_cuda_graph_vcycle_created[m_num_system_types] = {false,false};
    cudaGraph_t m_cuda_graph_vcycle[m_num_system_types] = {NULL,NULL};
    cudaGraphExec_t m_cuda_graph_exe_vcycle[m_num_system_types] = {NULL,NULL};
#endif
};

#if defined(AMREX_USE_GPU) || !defined(AMREX_USE_OMP)

using amrex::ParallelFor;

#else

// amrex::ParallelFor does not do OpenMP.  Thus we have hpmg::ParallelFor.

template <typename T, typename F>
void ParallelFor (T n, F&& f) noexcept
{
#pragma omp parallel for simd
    for (T i = 0; i < n; ++i) {
        f(i);
    }
}

template <typename F>
void ParallelFor (amrex::Box const& box, F&& f) noexcept
{
    const auto lo = amrex::lbound(box);
    const auto hi = amrex::ubound(box);
#pragma omp parallel for collapse(2)
    for (int k = lo.z; k <= hi.z; ++k) {
    for (int j = lo.y; j <= hi.y; ++j) {
#pragma omp simd
    for (int i = lo.x; i <= hi.x; ++i) {
        f(i,j,k);
    }}}
}

template <typename F>
void ParallelFor (amrex::Box const& box, int ncomp, F&& f) noexcept
{
    const auto lo = amrex::lbound(box);
    const auto hi = amrex::ubound(box);
#pragma omp parallel for collapse(3)
    for (int n = 0; n < ncomp; ++n) {
        for (int k = lo.z; k <= hi.z; ++k) {
        for (int j = lo.y; j <= hi.y; ++j) {
#pragma omp simd
        for (int i = lo.x; i <= hi.x; ++i) {
            f(i,j,k,n);
        }}}
    }
}

#endif


template <typename F>
void ParallelForCheckerboard (amrex::Box const& box, int ncomp, F&& f, int color=-1) noexcept
{
    const int len_x = box.length(0);
    const int len_y = box.length(1);

    AMREX_ALWAYS_ASSERT(box.length(2) == 1 && box.smallEnd() == amrex::IntVect::TheZeroVector());

    int npts = 0;
    int start = 0;
    if (color==-1) {
        npts = len_x * len_y;
    } else if (color%2==0) {
        npts = (len_x * len_y + 1)/2;
    } else {
        start = (len_x * len_y + 1)/2;
        npts = (len_x * len_y)/2;
    }

    ParallelFor(npts,
        [=] AMREX_GPU_DEVICE (int idx) noexcept {
            //const int n = idx / npts;
            idx += start ;//- n * npts;

            const int color_loc = 2 * idx >= len_x * len_y;

            const int idxn = 2 * idx - (len_x * len_y - !(len_y&1) ) * color_loc;

            const int j = idxn / len_x;
            int i = idxn - j * len_x;

            i += !(len_x&1)*( (j&1)*(1-2*color_loc)*!(len_y&1) + ((j&1)^color_loc)*(len_y&1) );

            for (int n=0; n<ncomp; ++n) f(i, j, n);
        });
}

}

#endif
