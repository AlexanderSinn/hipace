/* Copyright 2020-2022
 *
 * This file is part of HiPACE++.
 *
 * Authors: AlexanderSinn, Axel Huebl, MaxThevenet, Severin Diederichs
 *
 * License: BSD-3-Clause-LBNL
 */
#include "FFTPoissonSolverDirichletAMReX.H"
#include "fft/AnyFFT.H"
#include "fields/Fields.H"
#include "utils/Constants.H"
#include "utils/GPUUtil.H"
#include "utils/HipaceProfilerWrapper.H"

FFTPoissonSolverDirichletAMReX::FFTPoissonSolverDirichletAMReX (
    amrex::BoxArray const& ba,
    amrex::DistributionMapping const& dm,
    amrex::Geometry const& gm )
{
    HIPACE_PROFILE("FFTPoissonSolverDirichletAMReX::define()");
    m_stagingArea = amrex::MultiFab(ba, dm, 1, amrex::IntVect{1, 1, 0});

    m_poisson = std::make_unique<amrex::FFT::Poisson<>>(gm,
        amrex::Array<std::pair<amrex::FFT::Boundary,amrex::FFT::Boundary>,AMREX_SPACEDIM>{
        std::make_pair(amrex::FFT::Boundary::odd, amrex::FFT::Boundary::odd),
        std::make_pair(amrex::FFT::Boundary::odd, amrex::FFT::Boundary::odd),
        std::make_pair(amrex::FFT::Boundary::odd, amrex::FFT::Boundary::odd)});
}

void
FFTPoissonSolverDirichletAMReX::SolvePoissonEquation (amrex::MultiFab& lhs_mf)
{
    HIPACE_PROFILE("FFTPoissonSolverDirichletAMReX::SolvePoissonEquation()");

    m_poisson->solve(lhs_mf, m_stagingArea);
}
