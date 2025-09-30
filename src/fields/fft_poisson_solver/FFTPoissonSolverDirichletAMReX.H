/* Copyright 2020-2021
 *
 * This file is part of HiPACE++.
 *
 * Authors: AlexanderSinn, MaxThevenet, Severin Diederichs
 * License: BSD-3-Clause-LBNL
 */
#ifndef FFT_POISSON_SOLVER_DIRICHLET_AMREX_H_
#define FFT_POISSON_SOLVER_DIRICHLET_AMREX_H_

#include "fields/fft_poisson_solver/fft/AnyFFT.H"
#include "FFTPoissonSolver.H"

#include <AMReX_MultiFab.H>
#include <AMReX_GpuComplex.H>
#include <AMReX_FFT_Poisson.H>

/**
 * \brief This class handles functions and data to perform transverse Fourier-based Poisson solves.
 *
 * For a given source S, it solves equation Laplacian(F) = S and returns F.
 * Once an instance is created, a typical use consists in:
 * 1. Compute S directly in FFTPoissonSolver::m_stagingArea
 * 2. Call FFTPoissonSolver::SolvePoissonEquation(mf), which will solve Poisson equation with RHS
 *    in the staging area and return the LHS in mf.
 */
class FFTPoissonSolverDirichletAMReX final : public FFTPoissonSolver
{
public:
    /** Constructor */
    FFTPoissonSolverDirichletAMReX ( amrex::BoxArray const& a_realspace_ba,
                                      amrex::DistributionMapping const& dm,
                                      amrex::Geometry const& gm);

    /** virtual destructor */
    virtual ~FFTPoissonSolverDirichletAMReX () override final {}

    /**
     * Solve Poisson equation. The source term must be stored in the staging area m_stagingArea prior to this call.
     *
     * \param[in] lhs_mf Destination array, where the result is stored.
     */
    virtual void SolvePoissonEquation (amrex::MultiFab& lhs_mf) override final;

    /** Position and relative factor used to apply inhomogeneous Dirichlet boundary conditions */
    virtual amrex::Real BoundaryOffset() override final { return 1.; } // ?
    virtual amrex::Real BoundaryFactor() override final { return 1.; } // ?

private:

    std::unique_ptr<amrex::FFT::Poisson<>> m_poisson;
};

#endif
