/* Copyright 2021-2022
 *
 * This file is part of HiPACE++.
 *
 * Authors: Axel Huebl, MaxThevenet, Severin Diederichs
 * License: BSD-3-Clause-LBNL
 */
#include "SliceSort.H"
#include "utils/HipaceProfilerWrapper.H"
#include "Hipace.H"

void
shiftSlippedParticles (BeamParticleContainer& beam, const int slice, amrex::Geometry const& geom)
{
    HIPACE_PROFILE("shiftSlippedParticles()");

    amrex::removeInvalidParticles(beam.getBeamSlice(WhichBeamSlice::This));

    const amrex::Real dz = geom.CellSize(2);
    const amrex::Real dzi = geom.InvCellSize(2);
    const amrex::Real zeta_min = geom.ProbLo(2) + dz * (slice - geom.Domain().smallEnd(2));

    const amrex::Real dt = Hipace::GetInstance().m_dt;
    const amrex::Real t_max = Hipace::GetInstance().m_physical_time + dt;

    const amrex::Real dt_dzi = dt * dzi;
    const amrex::Real cutoff = t_max + zeta_min * dt_dzi;

    const auto num_stay = amrex::partitionParticles(beam.getBeamSlice(WhichBeamSlice::This),
        [=] AMREX_GPU_DEVICE (auto& ptd, int i) {
            const amrex::Real time = ptd.rdata(BeamIdx::t)[i];
            const amrex::Real zeta = ptd.pos(2, i);
            return time + zeta * dt_dzi < cutoff;
        });

    const auto num_move = beam.getBeamSlice(WhichBeamSlice::This).numParticles() - num_stay;

    beam.getBeamSlice(WhichBeamSlice::Next_t).resize(num_move);

    auto ptd_this = beam.getBeamSlice(WhichBeamSlice::This).getParticleTileData();
    auto ptd_next_t = beam.getBeamSlice(WhichBeamSlice::Next_t).getParticleTileData();

    amrex::ParallelFor(num_move,
        [=] AMREX_GPU_DEVICE (int i)
        {
            amrex::copyParticle(ptd_next_t, ptd_this, num_stay + i, i);
        });

    amrex::Gpu::streamSynchronize();
    beam.getBeamSlice(WhichBeamSlice::This).resize(num_stay);
}
