/* Copyright 2022
 *
 * This file is part of HiPACE++.
 *
 * Authors: AlexanderSinn
 * License: BSD-3-Clause-LBNL
 */
#ifndef SALAME_H_
#define SALAME_H_

#include "Hipace.H"
#include "utility"

/** Calculate new weight for this slice of SALAME beams and recompute the effected fields
 * \param[in] hipace pointer to Hipace instance
 * \param[in] n_iter the number of SALAME iterations to be done
 * \param[in] do_advance if the SALAME-only field should be computed exactly with plasma particles
 * \param[in,out] last_islice the slice index of the previous slice with SALAME
 * \param[in,out] overloaded if the SALAME beam was overloaded in the last slice
 * \param[in] current_N_level number of MR levels active on the current slice
 * \param[in] is_first_step if this is the first time step of the simulation
 * \param[in] islice slice index of the whole domain
 * \param[in] relative_tolerance relative error tolerance to finish SALAME iterations early
 */
void
SalameModule (Hipace* hipace, const int n_iter, const bool do_advance, int& last_islice,
              bool& overloaded, const int current_N_level, const bool is_first_step,
              const int islice, const amrex::Real relative_tolerance);

#endif //  SALAME_H_
