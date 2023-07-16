/* Copyright 2021
 *
 * This file is part of HiPACE++.
 *
 * Authors: AlexanderSinn, MaxThevenet
 * License: BSD-3-Clause-LBNL
 */
#include "BoxSort.H"
#include "particles/beam/BeamParticleContainer.H"
#include "utils/HipaceProfilerWrapper.H"

#include <AMReX_ParticleTransformation.H>

void BoxSorter::sortParticlesByBox (BeamParticleContainer& a_beam, const amrex::Geometry& a_geom)
{
    HIPACE_PROFILE("sortBeamParticlesByBox()");
    amrex::Gpu::streamSynchronize();

    const index_type np = a_beam.getBeamInitSlice().numParticles();
    auto ptd = a_beam.getBeamInitSlice().getParticleTileData();

    int num_boxes = a_geom.Domain().length(2);

    m_box_permutations.resize(np);
    m_box_counts.resize(0);
    m_box_counts.resize(num_boxes+1, 0);
    m_box_counts.assign(num_boxes+1, 0);
    m_box_offsets.resize(0);
    m_box_offsets.resize(num_boxes+1, 0);
    m_box_offsets.assign(num_boxes+1, 0);
    amrex::Gpu::HostVector<index_type> local_offsets (np);

    auto p_box_counts = m_box_counts.dataPtr();
    auto p_local_offsets = local_offsets.dataPtr();
    auto p_permutations = m_box_permutations.dataPtr();

    // Extract box properties
    const int lo_z = a_geom.Domain().smallEnd(2);
    const amrex::Real dzi = a_geom.InvCellSize(2);
    const amrex::Real plo_z = a_geom.ProbLo(2);

    for (index_type i=0; i<np; ++i) {
        int dst_box = static_cast<int>((ptd.pos(2, i) - plo_z) * dzi - lo_z);
        if (ptd.id(i) < 0) dst_box = num_boxes; // if pid is invalid, remove particle
        if (dst_box < 0 || dst_box > num_boxes) {
            // particle has left domain transversely, stick it at the end and invalidate
            dst_box = num_boxes;
            ptd.id(i) = -std::abs(ptd.id(i));
        }
        p_local_offsets[i] = (p_box_counts[dst_box])++;;
    }

    std::exclusive_scan(m_box_counts.begin(), m_box_counts.end(), m_box_offsets.begin(), 0);

    //m_max_counts = *std::max_element(m_box_counts.begin(), m_box_counts.end());

    auto p_box_offsets = m_box_offsets.dataPtr();

    for (index_type i=0; i<np; ++i) {
        int dst_box = static_cast<int>((ptd.pos(2, i) - plo_z) * dzi - lo_z);
        if (ptd.id(i) < 0) dst_box = num_boxes; // if pid is invalid, remove particle
        if (dst_box < 0 || dst_box > num_boxes) dst_box = num_boxes;
        p_permutations[p_local_offsets[i] + p_box_offsets[dst_box]] = i;
    }

    BeamTileInit tmp{};
    tmp.resize(np);

    auto ptd_tmp = tmp.getParticleTileData();


    /*for (index_type i=0; i<np; ++i) {
        const int idx_src = p_permutations[i];

        ptd_tmp.rdata(BeamIdx::x)[i] = ptd.rdata(BeamIdx::x)[idx_src];
        ptd_tmp.rdata(BeamIdx::y)[i] = ptd.rdata(BeamIdx::y)[idx_src];
        ptd_tmp.rdata(BeamIdx::z)[i] = ptd.rdata(BeamIdx::z)[idx_src];
        ptd_tmp.rdata(BeamIdx::w)[i] = ptd.rdata(BeamIdx::w)[idx_src];
        ptd_tmp.rdata(BeamIdx::ux)[i] = ptd.rdata(BeamIdx::ux)[idx_src];
        ptd_tmp.rdata(BeamIdx::uy)[i] = ptd.rdata(BeamIdx::uy)[idx_src];
        ptd_tmp.rdata(BeamIdx::uz)[i] = ptd.rdata(BeamIdx::uz)[idx_src];
        ptd_tmp.idata(BeamIdx::id)[i] = ptd.idata(BeamIdx::id)[idx_src];
    }*/

    amrex::ParallelFor(np,
        [=] AMREX_GPU_DEVICE (const index_type i) {
            const int idx_src = p_permutations[i];

            ptd_tmp.rdata(BeamIdx::x)[i] = ptd.rdata(BeamIdx::x)[idx_src];
            ptd_tmp.rdata(BeamIdx::y)[i] = ptd.rdata(BeamIdx::y)[idx_src];
            ptd_tmp.rdata(BeamIdx::z)[i] = ptd.rdata(BeamIdx::z)[idx_src];
            ptd_tmp.rdata(BeamIdx::w)[i] = ptd.rdata(BeamIdx::w)[idx_src];
            ptd_tmp.rdata(BeamIdx::ux)[i] = ptd.rdata(BeamIdx::ux)[idx_src];
            ptd_tmp.rdata(BeamIdx::uy)[i] = ptd.rdata(BeamIdx::uy)[idx_src];
            ptd_tmp.rdata(BeamIdx::uz)[i] = ptd.rdata(BeamIdx::uz)[idx_src];
            ptd_tmp.idata(BeamIdx::id)[i] = ptd.idata(BeamIdx::id)[idx_src];
        }
    );

    a_beam.getBeamInitSlice().swap(tmp);
}
