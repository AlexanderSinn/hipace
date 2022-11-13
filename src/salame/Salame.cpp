/* Copyright 2022
 *
 * This file is part of HiPACE++.
 *
 * Authors: AlexanderSinn
 * License: BSD-3-Clause-LBNL
 */
#include "Salame.H"
#include "utils/GPUUtil.H"
#include "utils/HipaceProfilerWrapper.H"

void
SalameModule (Hipace* hipace, const int n_iter, const int lev, const int step, const int islice,
              const int islice_local, const amrex::Vector<BeamBins>& beam_bin, const int ibox)
{
    HIPACE_PROFILE("SalameModule()");

    static int last_islice = -1;
    if (islice + 1 != last_islice) {
        hipace->m_fields.duplicate(lev, WhichSlice::Salame, {"Ez_target"},
                                        WhichSlice::This, {"Ez"});
        last_islice = islice;
    }

    for (int iter=0; iter<n_iter; ++iter) {

    hipace->m_multi_plasma.AdvanceParticles(hipace->m_fields, hipace->m_laser, hipace->Geom(lev),
                                            true, true, true, iter==0, lev);

    hipace->m_fields.duplicate(lev, WhichSlice::Salame, {"jx", "jy"},
                                    WhichSlice::Next, {"jx_beam", "jy_beam"});

    hipace->m_multi_plasma.DepositCurrent(hipace->m_fields, hipace->m_laser,
            WhichSlice::Salame, true, true, false, false, false, hipace->Geom(lev), lev);

    hipace->m_multi_plasma.ResetParticles(lev);

    hipace->m_fields.setVal(0., lev, WhichSlice::Salame, "Ez", "jz_beam", "Sy", "Sx");

    hipace->m_fields.SolvePoissonEz(hipace->Geom(), lev, islice, WhichSlice::Salame);

    hipace->m_fields.duplicate(lev, WhichSlice::Salame, {"Ez_no_salame"},
                                    WhichSlice::Salame, {"Ez"});

    hipace->m_multi_beam.DepositCurrentSlice(hipace->m_fields, hipace->Geom(), lev, step,
        islice_local, beam_bin, hipace->m_box_sorters, ibox, false, true, false, WhichSlice::Salame);

    SalameInitializeSxSyWithBeam(hipace, lev);

    hipace->ExplicitMGSolveBxBy(lev, WhichSlice::Salame);

    hipace->m_fields.setVal(0., lev, WhichSlice::Salame, "Ez", "jx", "jy");

    SalameGetJxJyFromBxBy(hipace, lev);

    hipace->m_fields.SolvePoissonEz(hipace->Geom(), lev, islice, WhichSlice::Salame);

    auto [W, W_total] = SalameGetW(hipace, lev);

    amrex::Print() << "Salame weight factor on slice " << islice << " is " << W
                   << " Total weight is " << W_total << '\n';

    SalameMultiplyBeamWeight(W, hipace, lev, islice_local, beam_bin, ibox);

    hipace->m_fields.setVal(0., lev, WhichSlice::This, "jz_beam", "Sy", "Sx");

    hipace->m_multi_beam.DepositCurrentSlice(hipace->m_fields, hipace->Geom(), lev, step,
        islice_local, beam_bin, hipace->m_box_sorters, ibox, false, true, false, WhichSlice::This);

    hipace->m_grid_current.DepositCurrentSlice(hipace->m_fields, hipace->Geom(lev), lev, islice);

    hipace->InitializeSxSyWithBeam(lev);

    hipace->m_multi_plasma.ExplicitDeposition(hipace->m_fields, hipace->m_laser,
                                              hipace->Geom(lev), lev);

    hipace->ExplicitMGSolveBxBy(lev, WhichSlice::This);

    }
};


void
SalameInitializeSxSyWithBeam (Hipace* hipace, const int lev)
{
    using namespace amrex::literals;

    amrex::MultiFab& slicemf = hipace->m_fields.getSlices(lev, WhichSlice::Salame);

    const amrex::Real dx = hipace->Geom(lev).CellSize(Direction::x);
    const amrex::Real dy = hipace->Geom(lev).CellSize(Direction::y);

    for ( amrex::MFIter mfi(slicemf, DfltMfiTlng); mfi.isValid(); ++mfi ){

        Array3<amrex::Real> const arr = slicemf.array(mfi);

        const int Sx = Comps[WhichSlice::Salame]["Sx"];
        const int Sy = Comps[WhichSlice::Salame]["Sy"];
        const int jzb = Comps[WhichSlice::Salame]["jz_beam"];

        const amrex::Real mu0 = hipace->m_phys_const.mu0;

        amrex::ParallelFor(mfi.tilebox(),
            [=] AMREX_GPU_DEVICE (int i, int j, int) noexcept
            {
                const amrex::Real dx_jzb = (arr(i+1,j,jzb)-arr(i-1,j,jzb))/(2._rt*dx);
                const amrex::Real dy_jzb = (arr(i,j+1,jzb)-arr(i,j-1,jzb))/(2._rt*dy);

                // sy, to compute Bx
                arr(i,j,Sy) =   mu0 * ( - dy_jzb);
                // sx, to compute By
                arr(i,j,Sx) = - mu0 * ( - dx_jzb);
            });
    }
}


void
SalameGetJxJyFromBxBy (Hipace* hipace, const int lev)
{
    using namespace amrex::literals;

    amrex::MultiFab& salame_slicemf = hipace->m_fields.getSlices(lev, WhichSlice::Salame);
    amrex::MultiFab& slicemf = hipace->m_fields.getSlices(lev, WhichSlice::This);

    const amrex::Real a1_times_dz = ( 1901._rt / 720._rt ) * hipace->Geom(lev).CellSize(Direction::z);

    for ( amrex::MFIter mfi(salame_slicemf, DfltMfiTlng); mfi.isValid(); ++mfi ){

        Array3<amrex::Real> const salame_arr = salame_slicemf.array(mfi);
        Array2<const amrex::Real> const chi_arr = slicemf.const_array(mfi, Comps[WhichSlice::This]["chi"]);

        const int Bx = Comps[WhichSlice::Salame]["Bx"];
        const int By = Comps[WhichSlice::Salame]["By"];
        const int jx = Comps[WhichSlice::Salame]["jx"];
        const int jy = Comps[WhichSlice::Salame]["jy"];

        const amrex::Real mu0 = hipace->m_phys_const.mu0;

        amrex::ParallelFor(mfi.tilebox(),
            [=] AMREX_GPU_DEVICE (int i, int j, int) noexcept
            {
                salame_arr(i,j,jx) =  a1_times_dz * chi_arr(i,j) * salame_arr(i,j,By) / mu0;
                salame_arr(i,j,jy) = -a1_times_dz * chi_arr(i,j) * salame_arr(i,j,Bx) / mu0;
            });
    }
}

std::pair<amrex::Real, amrex::Real>
SalameGetW (Hipace* hipace, const int lev)
{
    using namespace amrex::literals;

    amrex::Real sum_W = 0._rt;
    amrex::Real sum_jz = 0._rt;

    amrex::MultiFab& slicemf = hipace->m_fields.getSlices(lev, WhichSlice::Salame);

    for ( amrex::MFIter mfi(slicemf, DfltMfiTlng); mfi.isValid(); ++mfi ){
        amrex::ReduceOps<amrex::ReduceOpSum, amrex::ReduceOpSum> reduce_op;
        amrex::ReduceData<amrex::Real, amrex::Real> reduce_data(reduce_op);
        using ReduceTuple = typename decltype(reduce_data)::Type;
        Array3<amrex::Real> const arr = slicemf.array(mfi);

        const int Ez = Comps[WhichSlice::Salame]["Ez"];
        const int Ez_target = Comps[WhichSlice::Salame]["Ez_target"];
        const int Ez_no_salame = Comps[WhichSlice::Salame]["Ez_no_salame"];
        const int jz = Comps[WhichSlice::Salame]["jz_beam"];

        reduce_op.eval(mfi.tilebox(), reduce_data,
            [=] AMREX_GPU_DEVICE (int i, int j, int) noexcept -> ReduceTuple
            {
                if (arr(i,j,Ez) == 0._rt) return {0._rt, 0._rt};
                return {
                    arr(i,j,jz) * (arr(i,j,Ez_target) - arr(i,j,Ez_no_salame)) / arr(i,j,Ez),
                    arr(i,j,jz)
                };
            });
        auto res = reduce_data.value(reduce_op);
        sum_W += amrex::get<0>(res);
        sum_jz += amrex::get<1>(res);
    }
    return {(sum_W / sum_jz) + 1._rt,  sum_W + sum_jz};
}

void
SalameMultiplyBeamWeight (const amrex::Real W, Hipace* hipace, const int lev, const int islice,
                          const amrex::Vector<BeamBins>& bins, const int ibox)
{
    for (int i=0; i<(hipace->m_multi_beam.get_nbeams()); i++) {
        auto& beam = hipace->m_multi_beam.getBeam(i);

        if (!beam.m_do_salame) continue;

        const int box_offset = hipace->m_box_sorters[i].boxOffsetsPtr()[ibox];

        const auto& aos = beam.GetArrayOfStructs(); // For id
        const auto pos_structs = aos.begin() + box_offset;
        auto& soa = beam.GetStructOfArrays(); // For momenta and weights
        amrex::Real * const wp = soa.GetRealData(BeamIdx::w).data() + box_offset;

        BeamBins::index_type const * const indices = bins[i].permutationPtr();
        BeamBins::index_type const * const offsets = bins[i].offsetsPtrCpu();

        BeamBins::index_type cell_start = offsets[islice];
        BeamBins::index_type cell_stop = offsets[islice+1];

        int const num_particles = cell_stop-cell_start;

        amrex::ParallelFor(
            num_particles,
            [=] AMREX_GPU_DEVICE (long idx) {
                // Particles in the same slice must be accessed through the bin sorter
                const int ip = indices[cell_start+idx];
                // Skip invalid particles and ghost particles not in the last slice
                if (pos_structs[ip].id() < 0) return;

                wp[ip] *= W;
            });
    }
}
