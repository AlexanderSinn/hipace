/* Copyright 2020-2022
 *
 * This file is part of HiPACE++.
 *
 * Authors: AlexanderSinn, MaxThevenet, Severin Diederichs, WeiqunZhang
 * coulibaly-mouhamed
 * License: BSD-3-Clause-LBNL
 */
#include "Fields.H"
#include "fft_poisson_solver/FFTPoissonSolverPeriodic.H"
#include "fft_poisson_solver/FFTPoissonSolverDirichletDirect.H"
#include "fft_poisson_solver/FFTPoissonSolverDirichletExpanded.H"
#include "fft_poisson_solver/FFTPoissonSolverDirichletFast.H"
#include "fft_poisson_solver/FFTPoissonSolverDirichletQuick.H"
#include "fft_poisson_solver/MGPoissonSolverDirichlet.H"
#include "Hipace.H"
#include "OpenBoundary.H"
#include "utils/DeprecatedInput.H"
#include "utils/HipaceProfilerWrapper.H"
#include "utils/Constants.H"
#include "utils/GPUUtil.H"
#include "utils/InsituUtil.H"
#include "particles/particles_utils/ShapeFactors.H"
#ifdef HIPACE_USE_OPENPMD
#   include <openPMD/auxiliary/Filesystem.hpp>
#endif

using namespace amrex::literals;

void
Fields::ReadParameters (const int nlev)
{
    m_slices = decltype(m_slices)(nlev);

    amrex::ParmParse ppf("fields");
    DeprecatedInput("fields", "do_dirichlet_poisson", "poisson_solver", "");
    queryWithParser(ppf, "insitu_period", m_insitu_period.m_func_str);
    m_insitu_period.compile();
    m_insitu_file_prefix = Hipace::m_output_folder + "/insitu";
    const bool set_file_prefix = queryWithParser(ppf, "insitu_file_prefix", m_insitu_file_prefix);
    if (set_file_prefix) {
        amrex::Print() <<
            "It is recommended to use hipace.output_folder instead of fields.insitu_file_prefix\n";
    }
    queryWithParser(ppf, "do_symmetrize", m_do_symmetrize);
    DeprecatedInput("fields", "extended_solve",
                    "boundary.particle_lo and boundary.particle_hi", "", true);
    DeprecatedInput("fields", "open_boundary", "boundary.field = Open", "", true);
}

void
Fields::AllocData (
    int lev, amrex::Geometry const& geom, const amrex::BoxArray& slice_ba,
    const amrex::DistributionMapping& slice_dm)
{
    HIPACE_PROFILE("Fields::AllocData()");
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(slice_ba.size() == 1,
        "Parallel field solvers not supported yet");

    if (lev==0) {

        m_lev0_periodicity = geom.periodicity();

        // Need 1 extra guard cell transversally for transverse derivative
        int nguards_xy = (Hipace::m_depos_order_xy + 1) / 2 + 1;
        // Check the temperature deposition order, if enabled
        if (Hipace::m_deposit_temp_individual &&
            Hipace::m_temperature_depos_order > Hipace::m_depos_order_xy) {
            nguards_xy = (Hipace::m_temperature_depos_order + 1) / 2 + 1;
        }
        m_slices_nguards = amrex::IntVect{nguards_xy, nguards_xy, 0};

        m_explicit = Hipace::m_explicit;
        m_any_neutral_background = Hipace::GetInstance().m_multi_plasma.AnySpeciesNeutralizeBackground();
        const bool any_salame = Hipace::GetInstance().m_multi_beam.AnySpeciesSalame();


        int isl = WhichSlice::This;
        Comps[isl].multi_emplace(N_Comps, "jx", "jy", "rhomjz");
        Comps[isl].multi_emplace(N_Comps, "ExmBy", "EypBx", "Ez", "Bz", "ExpBy", "EymBx");

        isl = WhichSlice::Prev_t;
        Comps[isl].multi_emplace(N_Comps, "ExmBy", "EypBx", "Ez", "Bz", "ExpBy", "EymBx");

        isl = WhichSlice::Prev_z;
        Comps[isl].multi_emplace(N_Comps, "ExmBy", "EypBx");

        isl = WhichSlice::Init;
        Comps[isl].multi_emplace(N_Comps, "Ez_prev_z", "Bz_prev_z", "Ez_prev_z2", "Bz_prev_z2",
                                 "ExpBy_prev_z", "EymBx_prev_z");
    }

    // allocate memory for fields
    if (N_Comps != 0) {
        m_slices[lev].define(
            slice_ba, slice_dm, N_Comps, m_slices_nguards,
            amrex::MFInfo().SetArena(amrex::The_Arena()));
        m_slices[lev].setVal(0._rt);
    }

    // set default Poisson solver based on the platform and resolution
    const bool is_even = std::max(slice_ba[0].length(0), slice_ba[0].length(1)) % 2 == 0;
#ifdef AMREX_USE_GPU
    std::string poisson_solver_str = is_even ? "FFTDirichletQuick" : "FFTDirichletFast";
#else
    std::string poisson_solver_str = is_even ? "FFTDirichletDirectEven" : "FFTDirichletDirectOdd";
#endif
    amrex::ParmParse ppf("fields");
    queryWithParser(ppf, "poisson_solver", poisson_solver_str);

    // The Poisson solver operates on transverse slices only.
    // The constructor takes the BoxArray and the DistributionMap of a slice,
    // so the FFTPlans are built on a slice.
    if (poisson_solver_str == "FFTDirichletDirectEven"){
        m_poisson_solver.push_back(std::unique_ptr<FFTPoissonSolverDirichletDirect>(
            new FFTPoissonSolverDirichletDirect(getSlices(lev).boxArray(),
                                                getSlices(lev).DistributionMap(),
                                                geom, true)));
    } else if (poisson_solver_str == "FFTDirichletDirectOdd"){
        m_poisson_solver.push_back(std::unique_ptr<FFTPoissonSolverDirichletDirect>(
            new FFTPoissonSolverDirichletDirect(getSlices(lev).boxArray(),
                                                getSlices(lev).DistributionMap(),
                                                geom, false)));
    } else if (poisson_solver_str == "FFTDirichletExpanded"){
        m_poisson_solver.push_back(std::unique_ptr<FFTPoissonSolverDirichletExpanded>(
            new FFTPoissonSolverDirichletExpanded(getSlices(lev).boxArray(),
                                                  getSlices(lev).DistributionMap(),
                                                  geom)) );
    } else if (poisson_solver_str == "FFTDirichletFast"){
        m_poisson_solver.push_back(std::unique_ptr<FFTPoissonSolverDirichletFast>(
            new FFTPoissonSolverDirichletFast(getSlices(lev).boxArray(),
                                              getSlices(lev).DistributionMap(),
                                              geom)) );
    } else if (poisson_solver_str == "FFTDirichletQuick"){
        m_poisson_solver.push_back(std::unique_ptr<FFTPoissonSolverDirichletQuick>(
            new FFTPoissonSolverDirichletQuick(getSlices(lev).boxArray(),
                                               getSlices(lev).DistributionMap(),
                                               geom)) );
    } else if (poisson_solver_str == "FFTPeriodic") {
        m_poisson_solver.push_back(std::unique_ptr<FFTPoissonSolverPeriodic>(
            new FFTPoissonSolverPeriodic(getSlices(lev).boxArray(),
                                         getSlices(lev).DistributionMap(),
                                         geom))  );
    } else if (poisson_solver_str == "MGDirichlet") {
        m_poisson_solver.push_back(std::unique_ptr<MGPoissonSolverDirichlet>(
            new MGPoissonSolverDirichlet(getSlices(lev).boxArray(),
                                         getSlices(lev).DistributionMap(),
                                         geom))  );
    } else {
        amrex::Abort("Unknown poisson solver '" + poisson_solver_str +
            "', must be 'FFTDirichletDirectEven', 'FFTDirichletDirectOdd', 'FFTDirichletExpanded', "
            "'FFTDirichletFast', 'FFTDirichletQuick', 'FFTPeriodic' or 'MGDirichlet'");
    }

    if (lev == 0 && m_insitu_period.isNonZero()) {
#ifdef HIPACE_USE_OPENPMD
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(m_insitu_file_prefix !=
            Hipace::GetInstance().m_openpmd_writer.m_file_prefix,
            "Must choose a different field insitu file prefix compared to the full diagnostics");
#endif
        // Allocate memory for in-situ diagnostics
        m_insitu_rdata.resize(geom.Domain().length(2)*m_insitu_nrp, 0.);
        m_insitu_sum_rdata.resize(m_insitu_nrp, 0.);
    }
}

/** \brief inner version of derivative */
template<int dir>
struct derivative_inner {
    // captured variables for GPU
    Array2<amrex::Real const> array;
    amrex::Real dx_inv;

    // derivative of field in dir direction (x or y)
    AMREX_GPU_DEVICE amrex::Real operator() (int i, int j) const noexcept {
        constexpr bool is_x_dir = dir == Direction::x;
        constexpr bool is_y_dir = dir == Direction::y;
        return (array(i+is_x_dir,j+is_y_dir) - array(i-is_x_dir,j-is_y_dir)) * dx_inv;
    }
};

/** \brief inner version of derivative */
template<>
struct derivative_inner<Direction::z> {
    // captured variables for GPU
    Array2<amrex::Real const> array1;
    Array2<amrex::Real const> array2;
    amrex::Real dz_inv;

    // derivative of field in z direction
    AMREX_GPU_DEVICE amrex::Real operator() (int i, int j) const noexcept {
        return (array1(i,j) - array2(i,j)) * dz_inv;
    }
};

/** \brief derivative in x or y direction */
template<int dir>
struct derivative {
    // use brace initialization as constructor
    amrex::MultiFab f_view; // field to calculate its derivative
    const amrex::Geometry& geom; // geometry of field

    // use .array(mfi) like with amrex::MultiFab
    derivative_inner<dir> array (amrex::MFIter& mfi) const {
        return derivative_inner<dir>{f_view.array(mfi), 0.5_rt*geom.InvCellSize(dir)};
    }
};

/** \brief derivative in z direction. Use fields from previous and next slice */
template<>
struct derivative<Direction::z> {
    // use brace initialization as constructor
    amrex::MultiFab f_view1; // field on previous slice to calculate its derivative
    amrex::MultiFab f_view2; // field on next slice to calculate its derivative
    const amrex::Geometry& geom; // geometry of field

    // use .array(mfi) like with amrex::MultiFab
    derivative_inner<Direction::z> array (amrex::MFIter& mfi) const {
        return derivative_inner<Direction::z>{f_view1.array(mfi), f_view2.array(mfi),
            0.5_rt*geom.InvCellSize(Direction::z)};
    }
};

/** \brief inner version of interpolated_field_xy */
template<int interp_order_xy, class ArrayType>
struct interpolated_field_xy_inner {
    // captured variables for GPU
    ArrayType array;
    amrex::Real dx_inv;
    amrex::Real dy_inv;
    amrex::Real offset0;
    amrex::Real offset1;

    // interpolate field in x, y with interp_order_xy order transversely,
    // x and y must be inside field box
    template<class...Args> AMREX_GPU_DEVICE
    amrex::Real operator() (amrex::Real x, amrex::Real y, Args...args) const noexcept {

        const amrex::Real xmid = (x - offset0)*dx_inv;
        const amrex::Real ymid = (y - offset1)*dy_inv;

        amrex::Real field_value = 0._rt;
        for (int iy=0; iy<=interp_order_xy; iy++){
            for (int ix=0; ix<=interp_order_xy; ix++){
                auto [shape_y, j] = shape_factor<interp_order_xy>(ymid, iy);
                auto [shape_x, i] = shape_factor<interp_order_xy>(xmid, ix);
                field_value += shape_x * shape_y * array(i, j, args...);
            }
        }
        return field_value;
    }
};

/** \brief interpolate field in x, y with interp_order_xy order transversely,
 * x and y must be inside field box */
template<int interp_order_xy, class MfabType>
struct interpolated_field_xy {
    // use brace initialization as constructor
    MfabType mfab; // MultiFab type object of the field
    amrex::Geometry geom; // geometry of field

    // use .array(mfi) like with amrex::MultiFab
    auto array (amrex::MFIter& mfi) const {
        auto mfab_array = to_array2(mfab.array(mfi));
        return interpolated_field_xy_inner<interp_order_xy, decltype(mfab_array)>{
            mfab_array, geom.InvCellSize(0), geom.InvCellSize(1),
            GetPosOffset(0, geom, geom.Domain()), GetPosOffset(1, geom, geom.Domain())};
    }
};

/** \brief inner version of guarded_field_xy */
struct guarded_field_xy_inner {
    // captured variables for GPU
    Array3<amrex::Real const> array;
    int lox;
    int hix;
    int loy;
    int hiy;

    AMREX_GPU_DEVICE amrex::Real operator() (int i, int j, int n) const noexcept {
        if (lox <= i && i <= hix && loy <= j && j <= hiy) {
            return array(i,j,n);
        } else return 0._rt;
    }
};

/** \brief if indices are outside of the fields box zero is returned */
struct guarded_field_xy {
    // use brace initialization as constructor
    amrex::MultiFab& mfab; // field to be guarded (zero extended)

    // use .array(mfi) like with amrex::MultiFab
    guarded_field_xy_inner array (amrex::MFIter& mfi) const {
        const amrex::Box bx = mfab[mfi].box();
        return guarded_field_xy_inner{mfab.const_array(mfi), bx.smallEnd(Direction::x),
            bx.bigEnd(Direction::x), bx.smallEnd(Direction::y), bx.bigEnd(Direction::y)};
    }
};

/** \brief Calculates dst = factor_a*src_a + factor_b*src_b. src_a and src_b can be derivatives
 *
 * \param[in] dst destination
 * \param[in] factor_a factor before src_a
 * \param[in] src_a first source
 * \param[in] factor_b factor before src_b
 * \param[in] src_b second source
 */
template<class FVA, class FVB>
void
LinCombination (amrex::MultiFab dst,
                const amrex::Real factor_a, const FVA& src_a,
                const amrex::Real factor_b, const FVB& src_b)
{
#ifdef AMREX_USE_OMP
#pragma omp parallel
#endif
    for ( amrex::MFIter mfi(dst, DfltMfiTlng); mfi.isValid(); ++mfi ){
        const Array2<amrex::Real> dst_array = dst.array(mfi);
        const auto src_a_array = to_array2(src_a.array(mfi));
        const auto src_b_array = to_array2(src_b.array(mfi));
        amrex::ParallelFor(to2D(mfi.growntilebox()),
            [=] AMREX_GPU_DEVICE(int i, int j) noexcept
            {
                dst_array(i,j) = factor_a * src_a_array(i,j) + factor_b * src_b_array(i,j);
            });
    }
}

/** \brief Calculates dst = factor*src. src can be a derivative
 *
 * \param[in] dst destination
 * \param[in] factor factor before src_a
 * \param[in] src first source
 */
template<class FV>
void
Multiply (amrex::MultiFab dst, const amrex::Real factor, const FV& src)
{
#ifdef AMREX_USE_OMP
#pragma omp parallel
#endif
    for ( amrex::MFIter mfi(dst, DfltMfiTlng); mfi.isValid(); ++mfi ){
        const Array2<amrex::Real> dst_array = dst.array(mfi);
        const auto src_array = to_array2(src.array(mfi));
        amrex::ParallelFor(to2D(mfi.growntilebox()),
            [=] AMREX_GPU_DEVICE(int i, int j) noexcept
            {
                dst_array(i,j) = factor * src_array(i,j);
            });
    }
}

void
Fields::Copy (const int current_N_level, const int i_slice, DiagnosticData& fd,
              const amrex::Vector<amrex::Geometry>& field_geom, MultiLaser& multi_laser)
{
    HIPACE_PROFILE("Fields::Copy()");
    constexpr int depos_order_xy = 1;
    constexpr int depos_order_z = 1;
    constexpr int depos_order_offset = depos_order_z / 2 + 1;

    const amrex::Real poff_calc_z = GetPosOffset(2, field_geom[0], field_geom[0].Domain());
    const amrex::Real poff_diag_x = GetPosOffset(0, fd.m_geom_io, fd.m_geom_io.Domain());
    const amrex::Real poff_diag_y = GetPosOffset(1, fd.m_geom_io, fd.m_geom_io.Domain());
    const amrex::Real poff_diag_z = GetPosOffset(2, fd.m_geom_io, fd.m_geom_io.Domain());

    // Interpolation in z Direction, done as if looped over diag_fab not i_slice
    // Calculate to which diag_fab slices this slice could contribute
    const int i_slice_min = i_slice - depos_order_offset;
    const int i_slice_max = i_slice + depos_order_offset;
    const amrex::Real pos_slice_min = i_slice_min * field_geom[0].CellSize(2) + poff_calc_z;
    const amrex::Real pos_slice_max = i_slice_max * field_geom[0].CellSize(2) + poff_calc_z;
    int k_min = static_cast<int>(amrex::Math::round((pos_slice_min - poff_diag_z)
                                                          * fd.m_geom_io.InvCellSize(2)));
    const int k_max = static_cast<int>(amrex::Math::round((pos_slice_max - poff_diag_z)
                                                          * fd.m_geom_io.InvCellSize(2)));

    amrex::Box diag_box = fd.m_geom_io.Domain();
    if (!fd.m_integrate_along_z) {
        // Put contributions from i_slice to different diag_fab slices in GPU vector
        m_rel_z_vec.resize(k_max+1-k_min);
        for (int k=k_min; k<=k_max; ++k) {
            const amrex::Real pos = k * fd.m_geom_io.CellSize(2) + poff_diag_z;
            const amrex::Real mid_i_slice = (pos - poff_calc_z)*field_geom[0].InvCellSize(2);
            m_rel_z_vec[k-k_min] = 0;
            for (int i=0; i<=depos_order_z; ++i) {
                auto [shape_z, k_cell] = shape_factor<depos_order_z>(mid_i_slice, i);
                if (k_cell == i_slice) {
                    m_rel_z_vec[k-k_min] = shape_z;
                }
            }
        }

        // Optimization: don’t loop over diag_fab slices with 0 contribution
        int k_start = k_min;
        int k_stop = k_max;
        for (int k=k_min; k<=k_max; ++k) {
            if (m_rel_z_vec[k-k_min] == 0) ++k_start;
            else break;
        }
        for (int k=k_max; k>=k_min; --k) {
            if (m_rel_z_vec[k-k_min] == 0) --k_stop;
            else break;
        }
        diag_box.setSmall(2, amrex::max(diag_box.smallEnd(2), k_start));
        diag_box.setBig(2, amrex::min(diag_box.bigEnd(2), k_stop));
    } else {
        m_rel_z_vec.resize(1);
        const amrex::Real pos_z = i_slice * field_geom[0].CellSize(2) + poff_calc_z;
        if (fd.m_geom_io.ProbLo(2) <= pos_z && pos_z <= fd.m_geom_io.ProbHi(2)) {
            m_rel_z_vec[0] = field_geom[0].CellSize(2);
            k_min = 0;
        } else {
            return;
        }
    }
    if (diag_box.isEmpty()) return;
    const int field_lev = fd.m_base_diag_type == DiagnosticData::diag_type::field ? fd.m_level : 0;

    auto& slice_mf = m_slices[field_lev];
    auto slice_func = interpolated_field_xy<depos_order_xy,
        guarded_field_xy>{{slice_mf}, field_geom[field_lev]};
    auto& laser_mf = multi_laser.getSlices();
    auto laser_func = interpolated_field_xy<depos_order_xy,
        guarded_field_xy>{{laser_mf}, multi_laser.GetLaserGeom()};

    m_rel_z_vec.copyToDeviceAsync();

    // Finally actual kernel: Interpolation in x, y, z of zero-extended fields
    for (amrex::MFIter mfi(slice_mf, DfltMfi); mfi.isValid(); ++mfi) {
        const int *diag_comps = fd.m_comps_output_idx.data();
        const amrex::Real *rel_z_data = m_rel_z_vec.data();
        const amrex::Real dx = fd.m_geom_io.CellSize(0);
        const amrex::Real dy = fd.m_geom_io.CellSize(1);

        if (fd.m_base_diag_type == DiagnosticData::diag_type::field &&
            current_N_level > fd.m_level) {
            auto slice_array = slice_func.array(mfi);
            amrex::Array4<amrex::Real> diag_array = fd.m_F_real.array();
            const amrex::Real clight = get_phys_const().c;
            amrex::ParallelFor(diag_box, fd.m_nfields,
                [=] AMREX_GPU_DEVICE(int i, int j, int k, int n) noexcept
                {
                    const amrex::Real x = i * dx + poff_diag_x;
                    const amrex::Real y = j * dy + poff_diag_y;
                    const int m = n[diag_comps];
                    diag_array(i,j,k,n) += rel_z_data[k-k_min] * slice_array(x,y,m);
                });
        } else if (fd.m_base_diag_type == DiagnosticData::diag_type::laser &&
                   multi_laser.UseLaser(i_slice)) {
            auto laser_array = laser_func.array(mfi);
            amrex::Array4<amrex::GpuComplex<amrex::Real>> diag_array_laser = fd.m_F_complex.array();
            amrex::ParallelFor(diag_box, fd.m_nfields,
                [=] AMREX_GPU_DEVICE(int i, int j, int k, int n) noexcept
                {
                    const amrex::Real x = i * dx + poff_diag_x;
                    const amrex::Real y = j * dy + poff_diag_y;
                    const int m = n[diag_comps];
                    if (m == -1) { // real=|a^2|, imag=0
                        diag_array_laser(i,j,k,n) += amrex::GpuComplex<amrex::Real>{
                            rel_z_data[k-k_min] * abssq(
                                laser_array(x,y,WhichLaserSlice::n00j00_r),
                                laser_array(x,y,WhichLaserSlice::n00j00_i)),
                            amrex::Real(0)};
                    } else {
                        diag_array_laser(i,j,k,n) += amrex::GpuComplex<amrex::Real>{
                            rel_z_data[k-k_min] * laser_array(x,y,m),
                            rel_z_data[k-k_min] * laser_array(x,y,m+1)
                        };
                    }
                });
        }
    }

    // sync before m_rel_z_vec is written to again by the next Copy
    amrex::Gpu::streamSynchronize();
}

void
Fields::InitializeSlices (int lev, int islice, const amrex::Vector<amrex::Geometry>& geom)
{
    HIPACE_PROFILE("Fields::InitializeSlices()");

    setVal(0., lev, WhichSlice::This, "jx", "jy", "rhomjz");
}

void
Fields::ShiftSlices (int lev)
{
    HIPACE_PROFILE("Fields::ShiftSlices()");

    shift(lev, WhichSlice::Prev_z, WhichSlice::This, "ExmBy", "EypBx");
}

void
Fields::AddRhoIons (const int lev)
{
}

/** \brief Sets non zero Dirichlet Boundary conditions in RHS which is the source of the Poisson
 * equation: laplace LHS = RHS
 *
 * \param[in] RHS source of the Poisson equation: laplace LHS = RHS
 * \param[in] solver_size size of RHS/poisson solver (no tiling)
 * \param[in] geom geometry of of RHS/poisson solver
 * \param[in] offset shift boundary value by offset number of cells
 * \param[in] factor multiply the boundary_value by this factor
 * \param[in] boundary_value functional object (Real x, Real y) -> Real value_of_potential
 */
template<class Functional>
void
SetDirichletBoundaries (Array2<amrex::Real> RHS, const amrex::Box& solver_size,
                        const amrex::Geometry& geom, const amrex::Real offset,
                        const amrex::Real factor, const Functional& boundary_value)
{
}

void
Fields::SetBoundaryCondition (amrex::Vector<amrex::Geometry> const& geom, const int lev,
                              const int which_slice, std::string component,
                              amrex::MultiFab&& staging_area,
                              amrex::Real offset, amrex::Real factor)
{
}

void
Fields::LevelUpBoundary (amrex::Vector<amrex::Geometry> const& geom, const int lev,
                         const int which_slice, const std::string& component,
                         const amrex::IntVect outer_edge, const amrex::IntVect inner_edge)
{
}

void
Fields::LevelUp (amrex::Vector<amrex::Geometry> const& geom, const int lev,
                 const int which_slice, const std::string& component)
{
}

void
Fields::SolvePoissonPsiExmByEypBxEzBz (amrex::Vector<amrex::Geometry> const& geom,
                                       const int current_N_level)
{
}

void
Fields::SolveFields (amrex::Vector<amrex::Geometry> const& geom)
{
    HIPACE_PROFILE("SolveFields()");

    using namespace amrex::literals;

    const int ExmBy_this = Comps[WhichSlice::This]["ExmBy"];
    const int EypBx_this = Comps[WhichSlice::This]["EypBx"];
    const int Ez_this = Comps[WhichSlice::This]["Ez"];
    const int Bz_this = Comps[WhichSlice::This]["Bz"];
    const int ExpBy_this = Comps[WhichSlice::This]["ExpBy"];
    const int EymBx_this = Comps[WhichSlice::This]["EymBx"];

    const int jx_this = Comps[WhichSlice::This]["jx"];
    const int jy_this = Comps[WhichSlice::This]["jy"];
    const int rhomjz_this = Comps[WhichSlice::This]["rhomjz"];

    const int ExmBy_prev_t = Comps[WhichSlice::Prev_t]["ExmBy"];
    const int EypBx_prev_t = Comps[WhichSlice::Prev_t]["EypBx"];
    const int Ez_prev_t = Comps[WhichSlice::Prev_t]["Ez"];
    const int Bz_prev_t = Comps[WhichSlice::Prev_t]["Bz"];
    const int ExpBy_prev_t = Comps[WhichSlice::Prev_t]["ExpBy"];
    const int EymBx_prev_t = Comps[WhichSlice::Prev_t]["EymBx"];

    const int ExmBy_prev_z = Comps[WhichSlice::Prev_z]["ExmBy"];
    const int EypBx_prev_z = Comps[WhichSlice::Prev_z]["EypBx"];

    const auto pc = get_phys_const();
    const amrex::Real clight = pc.c;
    const amrex::Real mu0 = pc.mu0;
    const amrex::Real dx_inv = geom[0].InvCellSize(0);
    const amrex::Real dy_inv = geom[0].InvCellSize(1);
    const amrex::Real dzeta_inv = geom[0].InvCellSize(2);
    const amrex::Real dtau = Hipace::GetInstance().m_dt;
    const amrex::Real dtau_c_inv = 1 / (clight * dtau);

    for ( amrex::MFIter mfi(m_slices[0], DfltMfiTlng); mfi.isValid(); ++mfi ){

        const Array3<amrex::Real> slice_array = m_slices[0].array(mfi);
        const Array2<amrex::Real> staging_array = m_poisson_solver[0]->StagingArea().array(mfi);

        amrex::ParallelFor(to2D(mfi.growntilebox()),
            [=] AMREX_GPU_DEVICE(int i, int j) noexcept
            {
                staging_array(i,j) = (
                    dtau_c_inv * dtau_c_inv * slice_array(i,j,ExmBy_prev_t)
                    - dtau_c_inv * clight * mu0 * slice_array(i,j,jx_this)
                    + dtau_c_inv * clight * 0.5_rt * dy_inv * (slice_array(i,j+1,Bz_prev_t) - slice_array(i,j-1,Bz_prev_t))
                    + dtau_c_inv * dzeta_inv * 2._rt * slice_array(i,j,ExmBy_prev_z)
                    - dtau_c_inv * 0.5_rt * dx_inv * (slice_array(i+1,j,Ez_prev_t) - slice_array(i-1,j,Ez_prev_t))
                    - mu0 * clight * clight * 0.5_rt * dx_inv * (slice_array(i+1,j,rhomjz_this) - slice_array(i-1,j,rhomjz_this))
                );
            });
    }

    amrex::MultiFab lhs_ExmBy = getField(0, WhichSlice::This, "ExmBy");
    m_poisson_solver[0]->SolvePoissonEquation2(lhs_ExmBy, dtau_c_inv * (dtau_c_inv + 2*dzeta_inv));

    for ( amrex::MFIter mfi(m_slices[0], DfltMfiTlng); mfi.isValid(); ++mfi ){

        const Array3<amrex::Real> slice_array = m_slices[0].array(mfi);
        const Array2<amrex::Real> staging_array = m_poisson_solver[0]->StagingArea().array(mfi);

        amrex::ParallelFor(to2D(mfi.growntilebox()),
            [=] AMREX_GPU_DEVICE(int i, int j) noexcept
            {
                staging_array(i,j) = (
                    dtau_c_inv * dtau_c_inv * slice_array(i,j,EypBx_prev_t)
                    - dtau_c_inv * clight * mu0 * slice_array(i,j,jy_this)
                    - dtau_c_inv * clight * 0.5_rt * dx_inv * (slice_array(i+1,j,Bz_prev_t) - slice_array(i-1,j,Bz_prev_t))
                    + dtau_c_inv * dzeta_inv * 2._rt * slice_array(i,j,EypBx_prev_z)
                    - dtau_c_inv * 0.5_rt * dy_inv * (slice_array(i,j+1,Ez_prev_t) - slice_array(i,j+1,Ez_prev_t))
                    - mu0 * clight * clight * 0.5_rt * dy_inv * (slice_array(i,j+1,rhomjz_this) - slice_array(i,j+1,rhomjz_this))
                );
            });
    }

    amrex::MultiFab lhs_EypBx = getField(0, WhichSlice::This, "EypBx");
    m_poisson_solver[0]->SolvePoissonEquation2(lhs_EypBx, dtau_c_inv * (dtau_c_inv + 2*dzeta_inv));

    for ( amrex::MFIter mfi(m_slices[0], DfltMfiTlng); mfi.isValid(); ++mfi ){

        const Array3<amrex::Real> slice_array = m_slices[0].array(mfi);

        amrex::ParallelFor(to2D(mfi.growntilebox()),
            [=] AMREX_GPU_DEVICE(int i, int j) noexcept
            {
                slice_array(i,j,Ez_this) = (
                    slice_array(i,j,Ez_prev_t)
                    + dtau * clight * mu0 * clight * clight * slice_array(i,j,rhomjz_this)
                    - dtau * clight * 0.5_rt * dx_inv * (slice_array(i+1,j,ExmBy_this) - slice_array(i-1,j,ExmBy_this))
                    - dtau * clight * 0.5_rt * dy_inv * (slice_array(i,j+1,EypBx_this) - slice_array(i,j-1,EypBx_this))
                );
                slice_array(i,j,Bz_this) = (
                    slice_array(i,j,Bz_prev_t)
                    + dtau * 0.5_rt * dy_inv * (slice_array(i,j+1,ExmBy_this) - slice_array(i,j-1,ExmBy_this))
                    - dtau * 0.5_rt * dx_inv * (slice_array(i+1,j,EypBx_this) - slice_array(i-1,j,EypBx_this))
                );
            });

        amrex::ParallelFor(to2D(mfi.growntilebox()),
            [=] AMREX_GPU_DEVICE(int i, int j) noexcept
            {
                slice_array(i,j,ExpBy_this) = (
                    slice_array(i,j,ExpBy_prev_t)
                    - dtau * clight * mu0 * clight * slice_array(i,j,jx_this)
                    + dtau * clight * clight * 0.5_rt * dy_inv * (slice_array(i,j+1,Bz_this) - slice_array(i,j-1,Bz_this))
                    + dtau * clight * 0.5_rt * dx_inv * (slice_array(i+1,j,Ez_this) - slice_array(i-1,j,Ez_this))
                );
                slice_array(i,j,EymBx_this) = (
                    slice_array(i,j,EymBx_prev_t)
                    - dtau * clight * mu0 * clight * slice_array(i,j,jy_this)
                    - dtau * clight * clight * 0.5_rt * dx_inv * (slice_array(i+1,j,Bz_this) - slice_array(i-1,j,Bz_this))
                    + dtau * clight * 0.5_rt * dy_inv * (slice_array(i,j+1,Ez_this) - slice_array(i,j-1,Ez_this))
                );
            });
    }
}

void
Fields::SolvePoissonEz (amrex::Vector<amrex::Geometry> const& geom,
                        const int current_N_level, const int which_slice)
{
}

void
Fields::SolvePoissonBxBy (amrex::Vector<amrex::Geometry> const& geom,
                          const int current_N_level, const int which_slice)
{
}

void
Fields::SymmetrizeFields (int field_comp, const int lev, const int symm_x, const int symm_y)
{
}

void
Fields::EnforcePeriodic (const bool do_sum, std::vector<int>&& comp_idx)
{
}

void
Fields::InitialBfieldGuess (const amrex::Real relative_Bfield_error,
                            const amrex::Real predcorr_B_error_tolerance, const int lev)
{
}

void
Fields::MixAndShiftBfields (const amrex::Real relative_Bfield_error,
                            const amrex::Real relative_Bfield_error_prev_iter,
                            const amrex::Real predcorr_B_mixing_factor, const int lev)
{
}

void
Fields::InSituComputeDiags (int step, int islice, const amrex::Geometry& geom3D,
                            amrex::Real time, bool is_last_step)
{
}

void
Fields::InSituWriteToFile (int step, amrex::Real time, const amrex::Geometry& geom3D,
                           bool is_last_step)
{
}
