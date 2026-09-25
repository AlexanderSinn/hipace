/* Copyright 2020-2022
 *
 * This file is part of HiPACE++.
 *
 * Authors: AlexanderSinn, Andrew Myers, Axel Huebl, MaxThevenet
 * Remi Lehe, Severin Diederichs, WeiqunZhang, coulibaly-mouhamed
 *
 * License: BSD-3-Clause-LBNL
 */
#include "Hipace.H"
#include "utils/HipaceProfilerWrapper.H"
#include "particles/sorting/SliceSort.H"
#include "particles/sorting/BoxSort.H"
#include "salame/Salame.H"
#include "utils/DeprecatedInput.H"
#include "utils/IOUtil.H"
#include "utils/GPUUtil.H"
#include "particles/pusher/GetAndSetPosition.H"
#include "mg_solver/HpMultiGrid.H"
#include "fields/fft_poisson_solver/fft/AnyFFT.H"

#include <AMReX_ParmParse.H>
#include <AMReX_IntVect.H>
#include <AMReX_IOFormat.H>
#ifdef AMREX_USE_LINEAR_SOLVERS
#  include <AMReX_MLALaplacian.H>
#  include <AMReX_MLMG.H>
#endif

#include <algorithm>
#include <memory>

Hipace_early_init::Hipace_early_init (Hipace* instance)
{
    Hipace::m_instance = instance;

    Parser::addConstantsToParser();

    amrex::ParmParse pph("hipace");
    queryWithParser(pph ,"normalized_units", Hipace::m_normalized_units);
    if (Hipace::m_normalized_units) {
        m_phys_const = make_constants_normalized();
    } else {
        m_phys_const = make_constants_SI();
    }
    Parser::replaceAmrexParamsWithParser();
    Parser::DebugPrint();

    queryWithParser(pph, "do_device_synchronize", DO_DEVICE_SYNCHRONIZE);
    queryWithParser(pph, "depos_order_xy", m_depos_order_xy);
    queryWithParser(pph, "depos_order_z", m_depos_order_z);
    queryWithParser(pph, "depos_derivative_type", m_depos_derivative_type);
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(m_depos_order_xy != 0 || m_depos_derivative_type != 0,
                            "Analytic derivative with depos_order=0 would vanish");
    queryWithParser(pph, "output_folder", Hipace::m_output_folder);

    amrex::ParmParse pp_amr("amr");
    int max_level = 0;
    queryWithParser(pp_amr, "max_level", max_level);
    m_N_level = max_level + 1;
    queryWithParser(pph, "ignore_noncritical_warnings", m_ignore_noncritical_warnings);
    AnyFFT::setup();
}

Hipace_early_init::~Hipace_early_init ()
{
    AnyFFT::cleanup();
}

Hipace&
Hipace::GetInstance ()
{
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(m_instance, "instance has not been initialized yet");
    return *m_instance;
}

Hipace::Hipace () :
    Hipace_early_init(this)
{
    m_fields.ReadParameters(m_N_level);
    m_multi_beam.ReadParameters();
    m_multi_plasma.ReadParameters();
    m_adaptive_time_step.ReadParameters(m_multi_beam.get_nbeams());
    m_multi_laser.ReadParameters();
    m_grid_current.ReadParameters();
    m_grid_ionization.ReadParameters();
    m_diags.ReadParameters(m_N_level, m_multi_laser.UseLaser());
#ifdef HIPACE_USE_OPENPMD
    m_openpmd_writer.ReadParameters();
#endif
    ReadParameters();
}

void
Hipace::ReadParameters ()
{
    amrex::ParmParse pp;// Traditionally, max_step and stop_time do not have prefix.
    queryWithParser(pp, "max_step", m_max_step);

    bool use_previous_rng = false;
    queryWithParser(pp, "use_previous_rng", use_previous_rng);
    if (use_previous_rng) {
        amrex::ResetRandomSeed(
            amrex::ParallelDescriptor::NProcs()-amrex::ParallelDescriptor::MyProc(),
            (amrex::ParallelDescriptor::NProcs()-1-amrex::ParallelDescriptor::MyProc())*1234567ULL + 12345ULL);
    }

    int seed;
    if (queryWithParser(pp, "random_seed", seed)) amrex::ResetRandomSeed(seed, seed);

    amrex::ParmParse pph("hipace");

    std::string str_dt {""};
    queryWithParser(pph, "dt", str_dt);
    if (str_dt != "adaptive") {
        queryWithParser(pph, "dt", m_dt);
        m_max_time = std::copysign(m_max_time, m_dt);
    }
    queryWithParser(pph, "max_time", m_max_time);
    queryWithParser(pph, "verbose", m_verbose);
    m_numprocs = amrex::ParallelDescriptor::NProcs();
    if (m_ignore_noncritical_warnings) {
        if (m_numprocs > m_max_step + 1 && amrex::ParallelDescriptor::IOProcessor()) {
            amrex::OutStream()
                << "WARNING: Please use more or equal time steps than the number of MPI ranks\n";
        }
    } else {
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(m_numprocs <= m_max_step + 1,
            "Please use more or equal time steps than the number of MPI ranks");
    }
    queryWithParser(pph, "predcorr_B_error_tolerance", m_predcorr_B_error_tolerance);
    queryWithParser(pph, "predcorr_max_iterations", m_predcorr_max_iterations);
    queryWithParser(pph, "predcorr_B_mixing_factor", m_predcorr_B_mixing_factor);
    queryWithParser(pph, "do_beam_jx_jy_deposition", m_do_beam_jx_jy_deposition);
    queryWithParser(pph, "do_beam_jz_minus_rho", m_do_beam_jz_minus_rho);
    m_deposit_rho = m_diags.needsRho();
    queryWithParser(pph, "deposit_rho", m_deposit_rho);
    m_deposit_rho_individual = m_diags.needsRhoIndividual();
    queryWithParser(pph, "deposit_rho_individual", m_deposit_rho_individual);
    queryWithParser(pph, "deposit_n", m_deposit_n);
    queryWithParser(pph, "deposit_n_ion_levels", m_deposit_n_ion_levels);
    m_deposit_temp_individual = m_diags.needsTempIndividual();
    queryWithParser(pph, "deposit_temp_individual", m_deposit_temp_individual);
    queryWithParser(pph, "temperature_depos_order", m_temperature_depos_order);
    queryWithParser(pph, "interpolate_neutralizing_background",
        m_interpolate_neutralizing_background);
    bool do_mfi_sync = false;
    queryWithParser(pph, "do_MFIter_synchronize", do_mfi_sync);
    DfltMfi.SetDeviceSync(do_mfi_sync).UseDefaultStream();
    DfltMfiTlng.SetDeviceSync(do_mfi_sync).UseDefaultStream();
    if (amrex::TilingIfNotGPU()) {
        DfltMfiTlng.EnableTiling();
    }

    DeprecatedInput("hipace", "external_ExmBy_slope", "beams.external_E(x,y,z,t)", "", true);
    DeprecatedInput("hipace", "external_Ez_slope", "beams.external_E(x,y,z,t)", "", true);
    DeprecatedInput("hipace", "external_Ez_uniform", "beams.external_E(x,y,z,t)", "", true);
    DeprecatedInput("hipace", "external_E_uniform", "beams.external_E(x,y,z,t)", "", true);
    DeprecatedInput("hipace", "external_B_uniform","beams.external_B(x,y,z,t)", "", true);
    DeprecatedInput("hipace", "external_E_slope", "beams.external_E(x,y,z,t)", "", true);
    DeprecatedInput("hipace", "external_B_slope", "beams.external_B(x,y,z,t)", "", true);

    queryWithParser(pph, "salame_n_iter", m_salame_n_iter);
    queryWithParser(pph, "salame_do_advance", m_salame_do_advance);
    std::string salame_target_str = "Ez_initial";
    queryWithParser(pph, "salame_Ez_target(zeta,zeta_initial,Ez_initial)", salame_target_str);
    m_salame_target_func = makeFunctionWithParser<3>(salame_target_str, m_salame_parser,
                                                     {"zeta", "zeta_initial", "Ez_initial"});
    queryWithParser(pph, "salame_relative_tolerance", m_salame_relative_tolerance);

    std::string solver = "explicit";
    queryWithParser(pph, "bxby_solver", solver);
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
        solver == "predictor-corrector" ||
        solver == "explicit",
        "hipace.bxby_solver must be explicit or predictor-corrector");
    m_explicit = solver == "explicit" ? true : false;
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(m_explicit || !m_multi_beam.AnySpeciesSalame(),
        "Cannot use SALAME algorithm with predictor-corrector solver");
    queryWithParser(pph, "MG_tolerance_rel", m_MG_tolerance_rel);
    queryWithParser(pph, "MG_tolerance_abs", m_MG_tolerance_abs);
    queryWithParser(pph, "MG_verbose", m_MG_verbose);
    queryWithParser(pph, "use_amrex_mlmg", m_use_amrex_mlmg);
    queryWithParser(pph, "do_shared_depos", m_do_shared_depos);
    queryWithParser(pph, "do_tiling", m_do_tiling);
    queryWithParser(pph, "tile_size", m_tile_size);
#ifdef AMREX_USE_GPU
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(m_do_tiling==0, "Tiling must be turned off to run on GPU.");
#endif

    queryWithParser(pph, "background_density_SI", m_background_density_SI);
    DeprecatedInput("hipace", "comms_buffer_on_gpu", "comms_buffer.on_gpu", "", true);
    DeprecatedInput("hipace", "comms_buffer_max_leading_slices",
        "comms_buffer.max_leading_slices", "", true);
    DeprecatedInput("hipace", "comms_buffer_max_trailing_slices",
        "comms_buffer.max_trailing_slices)", "", true);

    DeprecatedInput("geometry", "is_periodic", "boundary.field and boundary.particle",
        "\n\n"
        "To directly replace geometry.is_periodic = 1 1 1 use:\n"
        "boundary.field = Periodic\n"
        "boundary.particle = Periodic\n"
        "However it's usually better to instead use:\n"
        "boundary.field = Dirichlet\n"
        "boundary.particle = Periodic\n"
        "or:\n"
        "boundary.field = Dirichlet\n"
        "boundary.particle = Reflecting\n"
        "\n"
        "To replace geometry.is_periodic = 0 0 0 use:\n"
        "boundary.field = Dirichlet\n"
        "boundary.particle = Absorbing\n", true);

    amrex::ParmParse ppb("boundary");
    std::string field_boundary = "";
    getWithParser(ppb, "field", field_boundary);
    if (field_boundary == "Dirichlet") {
        m_boundary_field = FieldBoundary::Dirichlet;
    } else if (field_boundary == "Periodic") {
        m_boundary_field = FieldBoundary::Periodic;
    } else if (field_boundary == "Open") {
        m_boundary_field = FieldBoundary::Open;
    } else {
        amrex::Abort("Unknown field boundary '" + field_boundary +
            "', must be 'Dirichlet', 'Periodic' or 'Open'");
    }

    std::string particle_boundary = "";
    getWithParser(ppb, "particle", particle_boundary);
    if (particle_boundary == "Reflecting") {
        m_boundary_particles = ParticleBoundary::Reflecting;
    } else if (particle_boundary == "Periodic") {
        m_boundary_particles = ParticleBoundary::Periodic;
    } else if (particle_boundary == "Absorbing") {
        m_boundary_particles = ParticleBoundary::Absorbing;
    } else {
        amrex::Abort("Unknown particle boundary '" + particle_boundary +
            "', must be 'Reflecting', 'Periodic' or 'Absorbing'");
    }

    MakeGeometry();

    m_boundary_particle_lo = {m_3D_geom[0].ProbLo(0), m_3D_geom[0].ProbLo(1)};
    m_boundary_particle_hi = {m_3D_geom[0].ProbHi(0), m_3D_geom[0].ProbHi(1)};
    queryWithParser(ppb, "particle_lo", m_boundary_particle_lo);
    queryWithParser(ppb, "particle_hi", m_boundary_particle_hi);
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
        m_boundary_particle_lo[0] >= m_3D_geom[0].ProbLo(0) &&
        m_boundary_particle_lo[1] >= m_3D_geom[0].ProbLo(1) &&
        m_boundary_particle_hi[0] <= m_3D_geom[0].ProbHi(0) &&
        m_boundary_particle_hi[1] <= m_3D_geom[0].ProbHi(1),
        "Particle boundary must be contained within the simulation domain");

    // use level 0 as default for laser geometry
    m_multi_laser.MakeLaserGeometry(m_3D_geom[0]);

    m_use_laser = m_multi_laser.UseLaser();

    queryWithParser(pph, "collisions", m_collision_names);
    /** Initialize the collision objects */
    m_ncollisions = m_collision_names.size();
    for (int i = 0; i < m_ncollisions; ++i) {
        m_all_collisions.emplace_back(CoulombCollision());
        m_all_collisions.back().ReadParameters(m_multi_plasma.m_names, m_multi_beam.m_names, m_collision_names[i]);
    }
    if (m_normalized_units && m_ncollisions > 0) {
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(m_background_density_SI!=0,
            "For collisions with normalized units, a background plasma density must "
            "be specified via 'hipace.background_density_SI'");
    }

    // external fields applied to the grid
    amrex::Array<std::string, 2> field_str = {"0", "0"};
    m_use_grid_external_fields = queryWithParser(pph, "grid_external_fields(x,y,z,t)", field_str);
    for (int i = 0; i < 2; ++i) {
        m_grid_external_fields[i] = makeFunctionWithParser<4>(field_str[i],
            m_grid_external_fields_parser[i], {"x", "y", "z", "t"});
    }
    DeprecatedInput("hipace", "grid_external_E(x,y,z,t)", "grid_external_fields(x,y,z,t)");
    DeprecatedInput("hipace", "grid_external_B(x,y,z,t)", "grid_external_fields(x,y,z,t)");
}

void
Hipace::InitData ()
{
    HIPACE_PROFILE("Hipace::InitData()");
#ifdef AMREX_USE_FLOAT
    amrex::Print() << "HiPACE++ (" << Hipace::Version() << ") running in single precision\n";
#else
    amrex::Print() << "HiPACE++ (" << Hipace::Version() << ") running in double precision\n";
#endif
#ifdef AMREX_USE_CUDA
    amrex::Print() << "using CUDA version " << __CUDACC_VER_MAJOR__ << "." << __CUDACC_VER_MINOR__
                   << "." << __CUDACC_VER_BUILD__ << "\n";
#endif

    for (int lev=0; lev<m_N_level; ++lev) {
        m_fields.AllocData(lev, m_3D_geom[lev], m_slice_ba[lev], m_slice_dm[lev]);
    }

    m_diags.Initialize(m_N_level, m_multi_laser.UseLaser());

    m_initial_time = m_multi_beam.InitData(m_3D_geom[0]);

    m_multi_buffer.initialize(m_3D_geom[0].Domain().length(2), m_multi_beam, m_fields);

    amrex::ParmParse pph("hipace");
    queryWithParser(pph, "initial_time", m_initial_time);
    queryWithParser(pph, "initial_step", m_initial_step);

    bool do_output_input = false;
    queryWithParser(pph, "output_input", do_output_input);
    if (do_output_input && amrex::ParallelDescriptor::IOProcessor()) {
        amrex::OutStream() <<
            "Input parameters:==================================================================\n";
        amrex::ParmParse::dumpTable(amrex::OutStream(), true);
        amrex::OutStream() <<
            "===================================================================================\n";
    }
}

void
Hipace::MakeGeometry ()
{
    using namespace amrex::literals;

    m_3D_geom.resize(m_N_level);
    m_3D_dm.resize(m_N_level);
    m_3D_ba.resize(m_N_level);
    m_slice_geom.resize(m_N_level);
    m_slice_dm.resize(m_N_level);
    m_slice_ba.resize(m_N_level);
    m_plasma_fine_patch.resize(m_N_level);

    // make 3D Geometry, BoxArray, DistributionMapping on level 0
    amrex::ParmParse pp_amr("amr");
    std::array<int, 3> n_cells {0, 0, 0};
    getWithParser(pp_amr, "n_cell", n_cells);
    const amrex::Box domain_3D{amrex::IntVect(0,0,0), n_cells.data()};
    const int is_periodic[3] {
        int(m_boundary_field == FieldBoundary::Periodic),
        int(m_boundary_field == FieldBoundary::Periodic),
        int(false)
    };

    // this will get prob_lo and prob_hi from the input file
    m_3D_geom[0].define(domain_3D, nullptr, amrex::CoordSys::cartesian, is_periodic);

    amrex::BoxList bl{domain_3D};
    amrex::Vector<int> procmap{amrex::ParallelDescriptor::MyProc()};
    m_3D_ba[0].define(bl);
    m_3D_dm[0].define(procmap);

    // make 3D Geometry, BoxArray, DistributionMapping on level >= 1
    for (int lev=1; lev<m_N_level; ++lev) {
        amrex::ParmParse pp_mrlev("mr_lev" + std::to_string(lev));

        // get n_cell in x and y direction, z direction is calculated from the patch size
        std::array<int, 2> n_cells_lev {0, 0};
        std::array<amrex::Real, 3> patch_lo_lev {0, 0, 0};
        std::array<amrex::Real, 3> patch_hi_lev {0, 0, 0};
        getWithParser(pp_mrlev, "n_cell", n_cells_lev);
        getWithParser(pp_mrlev, "patch_lo", patch_lo_lev);
        getWithParser(pp_mrlev, "patch_hi", patch_hi_lev);

        std::array<amrex::Real, 2> ref_ratio {0, 0}; // relative to level 0
        const bool rr_specified = queryWithParser(pp_mrlev, "ref_ratio", ref_ratio);

        m_plasma_fine_patch[lev] = {0, 0}; // relative to level lev patch length
        queryWithParser(pp_mrlev, "plasma_fine_patch", m_plasma_fine_patch[lev]);

        if (rr_specified) {
            std::array<amrex::Real, 2> patch_center_lev {
                0.5_rt * (patch_hi_lev[0] + patch_lo_lev[0]),
                0.5_rt * (patch_hi_lev[1] + patch_lo_lev[1])
            };

            std::array<amrex::Real, 2> patch_len_lev {
                n_cells_lev[0] * m_3D_geom[0].CellSize(0) / ref_ratio[0],
                n_cells_lev[1] * m_3D_geom[0].CellSize(1) / ref_ratio[1],
            };

            std::array<amrex::Real, 2> old_patch_len {
                patch_hi_lev[0] - patch_lo_lev[0],
                patch_hi_lev[1] - patch_lo_lev[1]
            };

            AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
                old_patch_len[0] > 0._rt && old_patch_len[1] > 0._rt &&
                (std::abs((patch_len_lev[0] - old_patch_len[0]) / old_patch_len[0]) <= 0.05_rt) &&
                (std::abs((patch_len_lev[1] - old_patch_len[1]) / old_patch_len[1]) <= 0.05_rt),
                "The refined patch would need to be changed by more than 5% "
                "to fit the requested refinement ratio! "
                "The patch length from patch_lo and patch_hi is " +
                amrex::ToString(old_patch_len) +
                " but the ref ratio and number of cells would give " +
                amrex::ToString(patch_len_lev) +
                "!"
            );

            patch_lo_lev[0] = patch_center_lev[0] - patch_len_lev[0] * 0.5_rt;
            patch_lo_lev[1] = patch_center_lev[1] - patch_len_lev[1] * 0.5_rt;

            patch_hi_lev[0] = patch_center_lev[0] + patch_len_lev[0] * 0.5_rt;
            patch_hi_lev[1] = patch_center_lev[1] + patch_len_lev[1] * 0.5_rt;
        }

        const amrex::Real pos_offset_z = GetPosOffset(2, m_3D_geom[0], m_3D_geom[0].Domain());

        const int zeta_lo = std::max( m_3D_geom[lev-1].Domain().smallEnd(2),
            int(amrex::Math::round((patch_lo_lev[2] - pos_offset_z) * m_3D_geom[0].InvCellSize(2)))
        );

        const int zeta_hi = std::min( m_3D_geom[lev-1].Domain().bigEnd(2),
            int(amrex::Math::round((patch_hi_lev[2] - pos_offset_z) * m_3D_geom[0].InvCellSize(2)))
        );

        patch_lo_lev[2] = (zeta_lo-0.5_rt)*m_3D_geom[0].CellSize(2) + pos_offset_z;
        patch_hi_lev[2] = (zeta_hi+0.5_rt)*m_3D_geom[0].CellSize(2) + pos_offset_z;

        const amrex::Box domain_3D_lev{amrex::IntVect(0,0,zeta_lo),
            amrex::IntVect(n_cells_lev[0]-1, n_cells_lev[1]-1, zeta_hi)};

        // non-periodic because it is internal
        m_3D_geom[lev].define(domain_3D_lev, amrex::RealBox(patch_lo_lev, patch_hi_lev),
                              amrex::CoordSys::cartesian, {0, 0, 0});

        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
            m_3D_geom[lev].ProbLo(0)-2*m_3D_geom[lev].CellSize(0)-2*m_3D_geom[lev-1].CellSize(0)
            >  m_3D_geom[lev-1].ProbLo(0) &&
            m_3D_geom[lev].ProbHi(0)+2*m_3D_geom[lev].CellSize(0)+2*m_3D_geom[lev-1].CellSize(0)
            <  m_3D_geom[lev-1].ProbHi(0) &&
            m_3D_geom[lev].ProbLo(1)-2*m_3D_geom[lev].CellSize(1)-2*m_3D_geom[lev-1].CellSize(1)
            >  m_3D_geom[lev-1].ProbLo(1) &&
            m_3D_geom[lev].ProbHi(1)+2*m_3D_geom[lev].CellSize(1)+2*m_3D_geom[lev-1].CellSize(1)
            <  m_3D_geom[lev-1].ProbHi(1),
            "Fine MR level must be fully nested inside the next coarsest level "
            "(with a few cells to spare)"
        );

        amrex::BoxList bl_lev{domain_3D_lev};
        amrex::Vector<int> procmap_lev{amrex::ParallelDescriptor::MyProc()};
        m_3D_ba[lev].define(bl_lev);
        m_3D_dm[lev].define(procmap_lev);
    }

    if (m_verbose > 0) {
        for (int lev=0; lev<m_N_level; ++lev) {
            amrex::Print()
                << "Using "
                << m_3D_geom[lev].Domain().length()
                << " cells\n    from "
                << amrex::RealVect{m_3D_geom[lev].ProbLoArray()}
                << "\n    to "
                << amrex::RealVect{m_3D_geom[lev].ProbHiArray()};
            if (lev > 0) {
                amrex::Print()
                    << "\n    on MR level "
                    << lev
                    << " with refinement ratio "
                    << amrex::RealVect{
                        m_3D_geom[0].CellSize(0) / m_3D_geom[lev].CellSize(0),
                        m_3D_geom[0].CellSize(1) / m_3D_geom[lev].CellSize(1),
                        m_3D_geom[0].CellSize(2) / m_3D_geom[lev].CellSize(2)
                    };
            }
            amrex::Print() << "\n";
        }
    }

    // make slice Geometry, BoxArray, DistributionMapping every level
    for (int lev=0; lev<m_N_level; ++lev) {
        amrex::Box slice_box = m_3D_geom[lev].Domain();
        slice_box.setSmall(2, 0);
        slice_box.setBig(2, 0);
        amrex::RealBox slice_realbox = m_3D_geom[lev].ProbDomain();
        slice_realbox.setLo(2, 0.);
        slice_realbox.setHi(2, m_3D_geom[lev].CellSize(2));

        m_slice_geom[lev].define(slice_box, slice_realbox, amrex::CoordSys::cartesian,
                                 m_3D_geom[lev].isPeriodic());
        m_slice_ba[lev].define(slice_box);
        m_slice_dm[lev].define(amrex::Vector<int>({amrex::ParallelDescriptor::MyProc()}));
    }
}

void
Hipace::Evolve ()
{
    HIPACE_PROFILE("Hipace::Evolve()");
    const double start_time = amrex::second();
    const int rank = amrex::ParallelDescriptor::MyProc();

    // now each rank starts with its own time step and writes to its own file. The first rank starts with step 0
    for (int step = m_initial_step + rank; step <= m_max_step; step += m_numprocs)
    {
        ResetAllQuantities();

        const amrex::Box& bx = m_3D_ba[0][0];

        m_physical_time = step == m_initial_step ? m_initial_time : m_multi_buffer.get_time();

        if (m_physical_time == std::numeric_limits<amrex::Real>::max()) {
            if (step+1 <= m_max_step && !m_has_last_step) {
                m_multi_buffer.put_time(m_physical_time);
            }
            break;
        }

        amrex::Real next_time = 0.;

        // adjust time step to reach max_time
        if (m_physical_time == m_max_time) {
            m_has_last_step = true;
            m_dt = 0.;
            next_time = std::numeric_limits<amrex::Real>::max();
        } else if ((m_physical_time + m_dt >= m_max_time && m_physical_time < m_max_time) ||
                   (m_physical_time + m_dt <= m_max_time && m_physical_time > m_max_time)) {
            m_dt = m_max_time - m_physical_time;
            next_time = m_max_time;
        } else {
            next_time = m_physical_time + m_dt;
        }

        if (m_verbose >= 1) {
            std::cout << utils::format_time{amrex::second() - start_time}
                      << " Rank " << rank
                      << " started step " << step
                      << " at time = " << m_physical_time
                      << " with dt = " << m_dt << std::endl;
        }

        if (step+1 <= m_max_step) {
            m_multi_buffer.put_time(next_time);
        }

        // need correct physical time for this
        const bool is_first_step = step == m_initial_step;
        const bool is_last_step = (step == m_max_step) || (m_physical_time == m_max_time);
        InitDiagnostics(step, m_physical_time, is_last_step);

        // Solve slices
        for (int isl = bx.bigEnd(Direction::z); isl >= bx.smallEnd(Direction::z); --isl){
            SolveOneSlice(isl, step, is_first_step, is_last_step);
        };

        WriteDiagnostics(step, m_physical_time, is_last_step);

        m_fields.InSituWriteToFile(step, m_physical_time, m_3D_geom[0], is_last_step);
        m_multi_beam.InSituWriteToFile(step, m_physical_time, m_3D_geom[0], is_last_step);
        m_multi_plasma.InSituWriteToFile(step, m_physical_time, m_3D_geom[0], is_last_step);
        m_multi_laser.InSituWriteToFile(step, m_physical_time, is_last_step);

        if (!m_explicit) {
            // averaging predictor corrector loop diagnostics
            m_predcorr_avg_iterations /= bx.length(Direction::z);
            m_predcorr_avg_B_error /= bx.length(Direction::z);
            if (m_verbose >= 2) {
                amrex::AllPrint() << "Rank " << rank
                                  << ": avg. number of iterations " << m_predcorr_avg_iterations
                                  <<" avg. transverse B field error " << m_predcorr_avg_B_error
                                  << "\n";
            }
            m_predcorr_avg_iterations = 0.;
            m_predcorr_avg_B_error = 0.;
        }

        FlushDiagnostics();
    }

    if (m_verbose >= 1) {
        // print total time, time per particle push and time per cell update
        amrex::ParallelDescriptor::ReduceRealSum(amrex::Vector<std::reference_wrapper<double>>{
            m_num_plasma_particles_pushed,
            m_num_beam_particles_pushed,
            m_num_field_cells_updated,
            m_num_laser_cells_updated
        }, HeadRankID());

        if (HeadRank()) {
            const double total_time_s = (amrex::second() - start_time);

            amrex::IOFormatSaver iofmtsaver(std::cout);
            std::cout << std::setprecision(4);

            std::cout << '\n' << "Finished Evolve after " << total_time_s << " seconds using "
                      << m_numprocs << (m_numprocs > 1 ? " ranks" : " rank" ) << std::endl;

            if (m_num_plasma_particles_pushed + m_num_beam_particles_pushed > 0.) {
                std::cout << "Total time per particle push: "
                          << 1e9 * total_time_s /
                            (m_num_plasma_particles_pushed + m_num_beam_particles_pushed)
                          << " nanoseconds";
                if (m_num_plasma_particles_pushed > 0. && m_num_beam_particles_pushed > 0.) {
                    std::cout << " ("
                              << 1e9 * total_time_s / m_num_plasma_particles_pushed << " plasma, "
                              << 1e9 * total_time_s / m_num_beam_particles_pushed << " beam)";
                }
                std::cout << std::endl;
            }

            if (m_num_field_cells_updated + m_num_laser_cells_updated > 0.) {
                std::cout << "Total time per cell update: "
                          << 1e9 * total_time_s /
                            (m_num_field_cells_updated + m_num_laser_cells_updated)
                          << " nanoseconds";
                if (m_num_field_cells_updated > 0. && m_num_laser_cells_updated > 0.) {
                    std::cout << " ("
                              << 1e9 * total_time_s / m_num_field_cells_updated << " field, "
                              << 1e9 * total_time_s / m_num_laser_cells_updated << " laser)";
                }
                std::cout << std::endl;
            }
        }
    }
}

void
Hipace::SolveOneSlice (int islice, int step, bool is_first_step, bool is_last_step)
{
    HIPACE_PROFILE("Hipace::SolveOneSlice()");

    m_num_field_cells_updated += m_slice_geom[0].Domain().d_numPts();

    m_fields.InitializeSlices(0, islice, m_3D_geom);

    m_multi_buffer.get_data(islice, m_multi_beam, m_fields, WhichBeamSlice::This);

    m_multi_beam.DepositCurrentSlice(m_fields, m_3D_geom, 0, islice);

    m_fields.SolveFields(m_3D_geom);

    FillBeamDiagnostics(step, m_physical_time, is_last_step);

    m_diags.FillDiagnostics(
        islice, 1,
        m_fields, m_multi_laser,
        m_multi_plasma, m_multi_beam,
        m_3D_geom
    );

    m_multi_beam.AdvanceBeamParticlesSlice(m_fields, m_3D_geom, islice, 1);

    m_multi_beam.shiftSlippedParticles(islice, m_3D_geom[0]);

    m_multi_buffer.put_data(islice, m_multi_beam, m_fields, WhichBeamSlice::Next_t, is_last_step);

    m_fields.ShiftSlices(0);
}

void
Hipace::ResetAllQuantities ()
{
    if (m_use_laser) {
        m_multi_laser.getSlices().setVal(0.);
    }

    for (int lev=0; lev<m_N_level; ++lev) {
        if (m_fields.getSlices(lev).nComp() != 0) {
            m_fields.getSlices(lev).setVal(0.);
        }
    }
}

void
Hipace::SetInitialConditions (const int islice)
{
    HIPACE_PROFILE("Hipace::SetInitialConditions()");
    if (!m_use_grid_external_fields) {
        m_fields.setVal(0., 0, WhichSlice::Prev_t, "ExmBy", "EypBx", "Ez", "Bz", "ExpBy", "EymBx");
        return;
    }

    using namespace amrex::literals;

    const amrex::Real dx = m_3D_geom[0].CellSize(Direction::x);
    const amrex::Real dy = m_3D_geom[0].CellSize(Direction::y);
    const amrex::Real dz = m_3D_geom[0].CellSize(Direction::z);

    const amrex::Real dx_inv = m_3D_geom[0].InvCellSize(Direction::x);
    const amrex::Real dy_inv = m_3D_geom[0].InvCellSize(Direction::y);
    const amrex::Real dz_inv = m_3D_geom[0].InvCellSize(Direction::z);

    const amrex::Real poff_x = GetPosOffset(0, m_3D_geom[0], m_3D_geom[0].Domain());
    const amrex::Real poff_y = GetPosOffset(1, m_3D_geom[0], m_3D_geom[0].Domain());
    const amrex::Real poff_z = GetPosOffset(2, m_3D_geom[0], m_3D_geom[0].Domain());

    auto external_fields = m_grid_external_fields;

    const int ExmBy_prev_t = Comps[WhichSlice::Prev_t]["ExmBy"];
    const int EypBx_prev_t = Comps[WhichSlice::Prev_t]["EypBx"];
    const int Ez_prev_t = Comps[WhichSlice::Prev_t]["Ez"];
    const int Bz_prev_t = Comps[WhichSlice::Prev_t]["Bz"];
    const int ExpBy_prev_t = Comps[WhichSlice::Prev_t]["ExpBy"];
    const int EymBx_prev_t = Comps[WhichSlice::Prev_t]["EymBx"];

    const int Ez_prev_z = Comps[WhichSlice::Init]["Ez_prev_z"];
    const int Bz_prev_z = Comps[WhichSlice::Init]["Bz_prev_z"];
    const int Ez_prev_z2 = Comps[WhichSlice::Init]["Ez_prev_z2"];
    const int Bz_prev_z2 = Comps[WhichSlice::Init]["Bz_prev_z2"];
    const int ExpBy_prev_z = Comps[WhichSlice::Init]["ExpBy_prev_z"];
    const int EymBx_prev_z = Comps[WhichSlice::Init]["EymBx_prev_z"];

    const amrex::Real time = m_physical_time;
    const amrex::Real clight_inv = 1._rt / get_phys_const().c;

    amrex::MultiFab& slicemf = m_fields.getSlices(0);

    for ( amrex::MFIter mfi(slicemf, DfltMfiTlng); mfi.isValid(); ++mfi ){

        amrex::Box const& gbx = mfi.growntilebox();
        amrex::Box const& bx = mfi.tilebox();

        Array3<amrex::Real> const arr = slicemf.array(mfi);
        amrex::ParallelFor(to2D(gbx),
            [=] AMREX_GPU_DEVICE (int i, int j) noexcept
            {
                const amrex::Real x = i * dx + poff_x;
                const amrex::Real y = j * dy + poff_y;
                const amrex::Real z = islice * dz + poff_z;

                const amrex::Real ExpByp = external_fields[0](x, y, z, time);
                const amrex::Real EymBxp = external_fields[1](x, y, z, time);

                arr(i, j, ExmBy_prev_t) = 0.;
                arr(i, j, EypBx_prev_t) = 0.;

                arr(i, j, ExpBy_prev_t) = ExpByp;
                arr(i, j, EymBx_prev_t) = EymBxp;
            });

        amrex::ParallelFor(to2D(bx),
            [=] AMREX_GPU_DEVICE (int i, int j) noexcept
            {
                arr(i, j, Ez_prev_t) = arr(i, j, Ez_prev_z2) + dz * (
                    dx_inv * (arr(i+1, j, ExpBy_prev_z) - arr(i-1, j, ExpBy_prev_z))
                    + dy_inv * (arr(i, j+1, EymBx_prev_z) - arr(i, j-1, EymBx_prev_z))
                );
                arr(i, j, Bz_prev_t) = arr(i, j, Bz_prev_z2) + dz * clight_inv * (
                    dy_inv * (arr(i, j+1, ExpBy_prev_z) - arr(i, j-1, ExpBy_prev_z))
                    - dx_inv * (arr(i+1, j, EymBx_prev_z) - arr(i-1, j, EymBx_prev_z))
                );
            });

        amrex::ParallelFor(to2D(gbx),
            [=] AMREX_GPU_DEVICE (int i, int j) noexcept
            {
                arr(i, j, Ez_prev_z2) = arr(i, j, Ez_prev_z);
                arr(i, j, Ez_prev_z) = arr(i, j, Ez_prev_t);

                arr(i, j, Bz_prev_z2) = arr(i, j, Bz_prev_z);
                arr(i, j, Bz_prev_z) = arr(i, j, Bz_prev_t);

                arr(i, j, ExpBy_prev_z) = arr(i, j, ExpBy_prev_t);

                arr(i, j, EymBx_prev_z) = arr(i, j, EymBx_prev_t);
            });
    }
}


void
Hipace::InitDiagnostics (const int step, const amrex::Real time, const bool is_last_step)
{
#ifdef HIPACE_USE_OPENPMD
    // need correct physical time for this check
    if (m_diags.hasAnyOutput(step, time, is_last_step)) {
        m_openpmd_writer.InitDiagnostics();
    }
    if (m_diags.hasBeamOutput(step, time, is_last_step)) {
        m_openpmd_writer.InitBeamData(m_multi_beam, getDiagBeamNames());
    }
#endif
    m_diags.ResizeFDiagFAB(m_3D_geom, m_multi_laser.GetLaserGeom(), step, time, is_last_step);
}

void
Hipace::FillBeamDiagnostics (const int step, const amrex::Real time, const bool is_last_step)
{
#ifdef HIPACE_USE_OPENPMD
    if (m_diags.hasBeamOutput(step, time, is_last_step)) {
        m_openpmd_writer.CopyBeams(m_multi_beam, getDiagBeamNames());
    }
#else
    amrex::ignore_unused(step, time, is_last_step);
#endif
}

void
Hipace::WriteDiagnostics (const int step, const amrex::Real time, const bool is_last_step)
{
#ifdef HIPACE_USE_OPENPMD
    if (m_diags.hasAnyFieldOutput(step, time, is_last_step)) {
        m_openpmd_writer.WriteFieldDiagnostics(m_diags.getDiagData(),
            m_multi_laser, m_physical_time, step);
    }

    if (m_diags.hasBeamOutput(step, time, is_last_step)) {
        m_openpmd_writer.WriteBeamDiagnostics(m_multi_beam, m_physical_time, step,
            getDiagBeamNames(), m_3D_geom);
    }
#else
    amrex::ignore_unused(step, time, is_last_step);
    amrex::Print()<<"WARNING: HiPACE++ compiled without openPMD support, the simulation has no I/O.\n";
#endif
}

void
Hipace::FlushDiagnostics ()
{
#ifdef HIPACE_USE_OPENPMD
    m_openpmd_writer.flush();
#endif
}
