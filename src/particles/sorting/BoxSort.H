/* Copyright 2021
 *
 * This file is part of HiPACE++.
 *
 * Authors: MaxThevenet
 * License: BSD-3-Clause-LBNL
 */
#ifndef HIPACE_BoxSort_H_
#define HIPACE_BoxSort_H_

#include <AMReX_MultiFab.H>
#include <AMReX_AmrParticles.H>
#include <AMReX_Particles.H>
#include <AMReX_AmrCore.H>

class BeamParticleContainer;

class BoxSorter
{
public:
    using index_type = unsigned int;

    void sortParticlesByBox (BeamParticleContainer& a_beam, const amrex::Geometry& a_geom);


    /** Number of particles in each box, stored on the cpu */
    amrex::Gpu::HostVector<index_type> m_box_counts;
    /** Index of the first particle in each box, stored on the cpu*/
    amrex::Gpu::HostVector<index_type> m_box_offsets;
    amrex::Gpu::PinnedVector<index_type> m_box_permutations;

    index_type m_max_counts = 0;
};

#endif // HIPACE_BoxSort_H_
