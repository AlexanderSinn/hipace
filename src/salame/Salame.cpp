/* Copyright 2022
 *
 * This file is part of HiPACE++.
 *
 * Authors: AlexanderSinn
 * License: BSD-3-Clause-LBNL
 */
#include "Salame.H"
#include "particles/particles_utils/FieldGather.H"
#include "utils/GPUUtil.H"
#include "utils/HipaceProfilerWrapper.H"
#include "utils/OMPUtil.H"

void
SalameModule (Hipace* hipace, const int n_iter, const bool do_advance, int& last_islice,
              bool& overloaded, const int current_N_level, const bool is_first_step,
              const int islice, const amrex::Real relative_tolerance)
{
}
