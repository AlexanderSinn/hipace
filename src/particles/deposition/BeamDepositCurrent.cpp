/* Copyright 2020-2022
 *
 * This file is part of HiPACE++.
 *
 * Authors: AlexanderSinn, Andrew Myers, MaxThevenet, Remi Lehe
 * Severin Diederichs
 * License: BSD-3-Clause-LBNL
 */
#include "BeamDepositCurrent.H"
#include "DepositionUtil.H"
#include "particles/beam/BeamParticleContainer.H"
#include "particles/particles_utils/ShapeFactors.H"
#include "fields/Fields.H"
#include "utils/Constants.H"
#include "utils/GPUUtil.H"
#include "utils/HipaceProfilerWrapper.H"
#include "Hipace.H"

#include <AMReX_DenseBins.H>

void
DepositCurrentSlice (BeamParticleContainer& beam, Fields& fields,
                     amrex::Vector<amrex::Geometry> const& gm, int const lev,
                     const int islice)
{
    HIPACE_PROFILE("DepositCurrentSlice_BeamParticleContainer()");

    using namespace amrex::literals;

    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(Hipace::m_depos_order_z == 0 || Hipace::m_depos_order_z == 2,
        "Only order 0 or 2 deposition is allowed for beam per-slice Ez interpolation");

    // Extract the fields currents
    // Extract FabArray for this box (because there is currently no transverse
    // parallelization, the index we want in the slice multifab is always 0.
    // Fix later.
    amrex::FArrayBox& isl_fab = fields.getSlices(lev)[0];

    // we deposit to the beam currents, because the explicit solver
    // requires sometimes just the beam currents
    // Do not access the field if the kernel later does not deposit into it,
    // the field might not be allocated. Use -1 as dummy component instead
    const int     jxb_cmp = Comps[WhichSlice::This]["jx"];
    const int     jyb_cmp = Comps[WhichSlice::This]["jy"];
    const int rhomjzb_cmp = Comps[WhichSlice::This]["rhomjz"];

    // Offset for converting positions to indexes
    amrex::Real const x_pos_offset = GetPosOffset(0, gm[lev], isl_fab.box());
    amrex::Real const y_pos_offset = GetPosOffset(1, gm[lev], isl_fab.box());

    PhysConst const phys_const = get_phys_const();

    // Extract box properties
    const amrex::Real dt = Hipace::GetInstance().m_dt;
    const amrex::Real dti = 1 / dt;
    const amrex::Real dxi = gm[lev].InvCellSize(0);
    const amrex::Real dyi = gm[lev].InvCellSize(1);
    const amrex::Real dz = gm[lev].CellSize(2);
    const amrex::Real dzi = gm[lev].InvCellSize(2);
    const amrex::Real invvol = dxi * dyi * dzi;
    const amrex::Real zeta_min = gm[lev].ProbLo(2) + dz * (islice - gm[lev].Domain().smallEnd(2));

    const amrex::Real clight = phys_const.c;
    const amrex::Real q = beam.m_charge;

    const amrex::Real hipace_next_time = Hipace::GetInstance().m_physical_time + dt;

    amrex::AnyCTO(
        // use compile-time options
        amrex::TypeList<amrex::CompileTimeOptions<0, 1, 2, 3>>{},
        {Hipace::m_depos_order_xy},
        // call deposition function
        // The three functions passed as arguments to this lambda
        // are defined below as the next arguments.
        [&](auto is_valid, auto get_cell, auto deposit){
            constexpr auto ctos = deposit.GetOptions();
            constexpr int depos_order = ctos[0];
            constexpr int stencil_size = depos_order + 1;
            SharedMemoryDeposition<stencil_size, stencil_size, true>(
                beam.getNumParticles(WhichBeamSlice::This), is_valid, get_cell, deposit,
                isl_fab.array(), isl_fab.box(),
                beam.getBeamSlice(WhichBeamSlice::This).getParticleTileData(),
                amrex::GpuArray<int, 0>{},
                amrex::GpuArray<int, 3>{jxb_cmp, jyb_cmp, rhomjzb_cmp});
        },
        // is_valid
        // return whether the particle is valid and should deposit
        [=] AMREX_GPU_DEVICE (int ip, auto ptd, auto /*depos_order*/)
        {
            return ptd.id(ip).is_valid();
        },
        // get_cell
        // return the lowest cell index that the particle deposits into
        [=] AMREX_GPU_DEVICE (int ip, auto ptd, auto depos_order) -> amrex::IntVectND<2>
        {
            const amrex::Real xmid = (ptd.pos(0, ip) - x_pos_offset)*dxi;
            const amrex::Real ymid = (ptd.pos(1, ip) - y_pos_offset)*dyi;

            // --- Compute shape factors
            auto [shape_y, j] = shape_factor<depos_order>(ymid, 0);
            auto [shape_x, i] = shape_factor<depos_order>(xmid, 0);

            return {i, j};
        },
        // deposit
        // deposit the charge / current of one particle
        [=] AMREX_GPU_DEVICE (int ip, auto ptd,
                              Array3<amrex::Real> arr,
                              auto /*cache_idx*/, auto depos_idx,
                              auto depos_order)
        {
            const amrex::Real xp = ptd.pos(0, ip);
            const amrex::Real yp = ptd.pos(1, ip);
            const amrex::Real zp = ptd.pos(2, ip);

            const amrex::Real ux = ptd.rdata(BeamIdx::ux)[ip];
            const amrex::Real uy = ptd.rdata(BeamIdx::uy)[ip];
            const amrex::Real uz = ptd.rdata(BeamIdx::uz)[ip];

            const amrex::Real gaminv = amrex::Math::rsqrt(1.0_rt + ux*ux + uy*uy + uz*uz);

            const amrex::Real betax = ux*gaminv;
            const amrex::Real betay = uy*gaminv;
            const amrex::Real betaz = uz*gaminv;

            const amrex::Real time = ptd.rdata(BeamIdx::t)[ip];

            amrex::Real next_time = time + (zp - zeta_min) / (clight*(1._rt - betaz));
            next_time = std::min(next_time, hipace_next_time);

            const amrex::Real pdt = next_time - time;
            ptd.rdata(BeamIdx::dt)[ip] = pdt;
            const amrex::Real weight_factor = pdt * dti;

            const amrex::Real wq = q * ptd.rdata(BeamIdx::w)[ip] * invvol * weight_factor;

            // wqx, wqy wqz are particle current in each direction
            const amrex::Real wqx = clight*wq*betax;
            const amrex::Real wqy = clight*wq*betay;
            const amrex::Real wqrhomjz = wq*(1._rt - betaz);

            const amrex::Real xmid = (xp - x_pos_offset)*dxi;
            const amrex::Real ymid = (yp - y_pos_offset)*dyi;

            // Deposit current into jx, jy, jz, rhomjz
            for (int iy=0; iy<=depos_order; iy++){
                for (int ix=0; ix<=depos_order; ix++){

                    // --- Compute shape factors
                    auto [shape_y, j] = shape_factor<depos_order>(ymid, iy);
                    auto [shape_x, i] = shape_factor<depos_order>(xmid, ix);

                    amrex::Gpu::Atomic::Add(
                        arr.ptr(i, j, depos_idx[0]), shape_x * shape_y * wqx);
                    amrex::Gpu::Atomic::Add(
                        arr.ptr(i, j, depos_idx[1]), shape_x * shape_y * wqy);
                    amrex::Gpu::Atomic::Add(
                        arr.ptr(i, j, depos_idx[2]), shape_x * shape_y * wqrhomjz);
                }
            }
        });
}
