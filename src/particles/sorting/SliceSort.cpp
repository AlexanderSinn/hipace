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
shiftSlippedParticles (BeamParticleContainer& beam, const int slice, amrex::Geometry const& geom,
    const int beam_slice_src, const int beam_slice_dst)
{
    if (beam.getNumParticlesIncludingSlipped(beam_slice_src) == 0) {
        // nothing to do
        return;
    }

    HIPACE_PROFILE("shiftSlippedParticles()");

    // remove all invalid particles from beam_slice_src (including slipped)
    amrex::removeInvalidParticles(beam.getBeamSlice(beam_slice_src));

    // min_z is the lower end of beam_slice_src
    const amrex::Real min_z = geom.ProbLo(2) +
        (slice+(beam_slice_src-WhichBeamSlice::This)-geom.Domain().smallEnd(2))*geom.CellSize(2);

    // put non slipped particles at the start of the slice
    const int num_stay = amrex::partitionParticles(beam.getBeamSlice(beam_slice_src),
        [=] AMREX_GPU_DEVICE (auto& ptd, int i) {
            return ptd.pos(2, i) >= min_z;
        });

    const int num_slipped = beam.getBeamSlice(beam_slice_src).size() - num_stay;

    if (num_slipped == 0) {
        // nothing to do
        beam.resize(beam_slice_src, num_stay, 0);
        return;
    }

    const int next_size = beam.getNumParticles(beam_slice_dst);

    // there shouldn't be any slipped particles already on beam_slice_dst
    AMREX_ALWAYS_ASSERT(beam.getNumParticlesIncludingSlipped(beam_slice_dst) == next_size);

    beam.resize(beam_slice_dst, next_size, num_slipped);

    const auto ptd_src = beam.getBeamSlice(beam_slice_src).getParticleTileData();
    const auto ptd_dst = beam.getBeamSlice(beam_slice_dst).getParticleTileData();

    amrex::ParallelFor(num_slipped,
        [=] AMREX_GPU_DEVICE (int i)
        {
            // copy particles from beam_slice_src to beam_slice_dst
            amrex::copyParticle(ptd_dst, ptd_src, num_stay + i, next_size + i);
        });


    // stream sync before beam_slice_src is resized
    amrex::Gpu::streamSynchronize();

    beam.resize(beam_slice_src, num_stay, 0);
}
