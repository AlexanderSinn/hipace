
/* Copyright 2020-2022
 *
 * This file is part of HiPACE++.
 *
 * Authors: AlexanderSinn, Andrew Myers, MaxThevenet, Severin Diederichs
 *
 * License: BSD-3-Clause-LBNL
 */
#include "BeamParticleAdvance.H"
#include "ExternalFields.H"
#include "particles/particles_utils/FieldGather.H"
#include "utils/Constants.H"
#include "GetAndSetPosition.H"
#include "utils/HipaceProfilerWrapper.H"
#include "utils/GPUUtil.H"
#include "utils/OMPUtil.H"


struct MRLevelData {
    // Array to access fields
    Array3<const amrex::Real> slice_arr;
    // Properties associated with physical size of the box
    amrex::Real dx_inv = 0;
    amrex::Real dy_inv = 0;
    // Offset for converting positions to indexes
    amrex::Real x_pos_offset = 0;
    amrex::Real y_pos_offset = 0;
};


void
AdvanceBeamParticlesSlice (
    BeamParticleContainer& beam, const Fields& fields, amrex::Vector<amrex::Geometry> const& gm,
    const int slice, int const current_N_level)
{
    HIPACE_PROFILE("AdvanceBeamParticlesSlice()");
    using namespace amrex::literals;

    const PhysConst phys_const = get_phys_const();

    const amrex::Real time = Hipace::GetInstance().m_physical_time;
    const amrex::Real dt = Hipace::GetInstance().m_dt;

    const int ExmBy_this = Comps[WhichSlice::This]["ExmBy"];
    const int EypBx_this = Comps[WhichSlice::This]["EypBx"];
    const int Ez_this = Comps[WhichSlice::This]["Ez"];
    const int Bz_this = Comps[WhichSlice::This]["Bz"];
    const int ExpBy_this = Comps[WhichSlice::This]["ExpBy"];
    const int EymBx_this = Comps[WhichSlice::This]["EymBx"];

    const int lev0_idx = 0;

    // Extract field array from FabArrays in MultiFabs.
    // (because there is currently no transverse parallelization, the index
    // we want in the slice multifab is always 0. Fix later.
    const amrex::FArrayBox& slice_fab_lev0 = fields.getSlices(lev0_idx)[0];

    const MRLevelData level0data {
        slice_fab_lev0.const_array(),
        gm[lev0_idx].InvCellSize(0), gm[lev0_idx].InvCellSize(1),
        GetPosOffset(0, gm[lev0_idx], slice_fab_lev0.box()),
        GetPosOffset(1, gm[lev0_idx], slice_fab_lev0.box())
    };

    // Extract particle properties
    const auto ptd = beam.getBeamSlice(WhichBeamSlice::This).getParticleTileData();

    const auto enforceBC = EnforceBC();

    const amrex::Real clight = phys_const.c;
    const amrex::Real inv_clight = 1.0_rt/phys_const.c;
    const amrex::Real charge_mass_ratio = beam.m_charge / beam.m_mass;
    const amrex::Real min_z = gm[0].ProbLo(2) + (slice-gm[0].Domain().smallEnd(2))*gm[0].CellSize(2);

    // don't include slipped particles in count as they were already pushed
    Hipace::m_num_beam_particles_pushed += double(beam.getNumParticles(WhichBeamSlice::This));

    // Use OMP ParallelFor to use multiple threads when running on CPU
    omp::ParallelFor(
        amrex::TypeList<
            amrex::CompileTimeOptions<0, 1, 2, 3>
        >{}, {
            Hipace::m_depos_order_xy
        },
        beam.getNumParticles(WhichBeamSlice::This),
        [=] AMREX_GPU_DEVICE (int ip, auto depos_order) {

            if (!ptd.id(ip).is_valid()) return;

            amrex::Real xp = ptd.pos(0, ip);
            amrex::Real yp = ptd.pos(1, ip);
            amrex::Real zp = ptd.pos(2, ip);

            amrex::Real ux = ptd.rdata(BeamIdx::ux)[ip];
            amrex::Real uy = ptd.rdata(BeamIdx::uy)[ip];
            amrex::Real uz = ptd.rdata(BeamIdx::uz)[ip];

            amrex::Real gammap_inv = amrex::Math::rsqrt(1.0_rt + ux*ux + uy*uy + uz*uz);

            amrex::Real time = ptd.rdata(BeamIdx::t)[ip];
            const amrex::Real pdt = ptd.rdata(BeamIdx::dt)[ip];

            xp += pdt * clight * 0.5_rt * gammap_inv * ux;
            yp += pdt * clight * 0.5_rt * gammap_inv * uy;

            if (enforceBC(ptd, ip, xp, yp, ux, uy)) return;

            MRLevelData level_data = level0data;
            const auto [slice_arr, dx_inv, dy_inv, x_pos_offset, y_pos_offset] = level_data;

            amrex::Real ExmByp = 0._rt, EypBxp = 0._rt, Ezp = 0._rt;
            amrex::Real Bzp = 0._rt, ExpByp = 0._rt, EymBxp = 0._rt;

            doGatherShapeN<depos_order.value>(xp, yp, slice_arr,
                ExmByp, EypBxp, Ezp, Bzp, ExpByp, EymBxp,
                ExmBy_this, EypBx_this, Ez_this, Bz_this, ExpBy_this, EymBx_this,
                dx_inv, dy_inv, x_pos_offset, y_pos_offset);

            ExmByp *= inv_clight;
            EypBxp *= inv_clight;
            Ezp *= inv_clight;
            ExpByp *= inv_clight;
            EymBxp *= inv_clight;

            constexpr int nsub = 16;
            for (int isub=0; isub<nsub; ++isub) {
                const amrex::Real beta_x = ux * gammap_inv;
                const amrex::Real beta_y = uy * gammap_inv;
                const amrex::Real beta_z = uz * gammap_inv;

                const amrex::Real dt_ux = charge_mass_ratio * (
                    0.5 * ExpByp * (1._rt - beta_z)
                    + 0.5 * ExmByp * (1._rt + beta_z)
                    + beta_y * Bzp
                );
                const amrex::Real dt_uy = charge_mass_ratio * (
                    0.5 * EymBxp * (1._rt - beta_z)
                    + 0.5 * EypBxp * (1._rt + beta_z)
                    - beta_x * Bzp
                );
                const amrex::Real dt_uz = charge_mass_ratio * (
                    Ezp
                    + beta_x * (ExpByp - ExmByp)
                    - beta_y * (EypBxp - EymBxp)
                );

                ux += dt_ux * pdt * (1._rt / nsub);
                uy += dt_uy * pdt * (1._rt / nsub);
                uz += dt_uz * pdt * (1._rt / nsub);

                gammap_inv = amrex::Math::rsqrt(1.0_rt + ux*ux + uy*uy + uz*uz);
            }

            xp += pdt * clight * 0.5_rt * gammap_inv * ux;
            yp += pdt * clight * 0.5_rt * gammap_inv * uy;
            zp += pdt * clight * (gammap_inv * uz - 1._rt);
            time += pdt;

            ptd.pos(0, ip) = xp;
            ptd.pos(1, ip) = yp;
            ptd.pos(2, ip) = zp;
            ptd.rdata(BeamIdx::ux)[ip] = ux;
            ptd.rdata(BeamIdx::uy)[ip] = uy;
            ptd.rdata(BeamIdx::uz)[ip] = uz;
            ptd.rdata(BeamIdx::t)[ip] = time;
        });
}
