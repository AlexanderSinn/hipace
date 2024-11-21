/* Copyright 2021-2022
 *
 * This file is part of HiPACE++.
 *
 * Authors: MaxThevenet, Severin Diederichs
 * License: BSD-3-Clause-LBNL
 */
#ifndef HIPACE_SLICESORT_H_
#define HIPACE_SLICESORT_H_

#include "particles/beam/BeamParticleContainer.H"
#include "BoxSort.H"

#include <AMReX_MultiFab.H>

/** \brief move particles that are not in WhichBeamSlice::This (including slipped) anymore
 * to WhichBeamSlice::Next (slipped region) while also fully deleting invalid particles.
 *
 * \param[in] beam Beam particle container
 * \param[in] slice longitudinal slice
 * \param[in] geom Geometry
 * \param[in] beam_slice_src source beam slice
 * \param[in] beam_slice_dst destination beam slice
 */
void
shiftSlippedParticles (BeamParticleContainer& beam, const int slice, amrex::Geometry const& geom,
    const int beam_slice_src, const int beam_slice_dst);

#endif // HIPACE_SLICESORT_H_
