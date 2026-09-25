/* Copyright 2025
 *
 * This file is part of HiPACE++.
 *
 * Authors: AlexanderSinn, EyaDammak
 * License: BSD-3-Clause-LBNL
 */
#include "TemperatureDeposition.H"
#include "DepositionUtil.H"
#include "particles/particles_utils/ShapeFactors.H"
#include "particles/particles_utils/FieldGather.H"
#include "particles/plasma/PlasmaParticleContainer.H"
#include "fields/Fields.H"
#include "utils/Constants.H"
#include "Hipace.H"
#include "utils/HipaceProfilerWrapper.H"
#include "utils/Constants.H"
#include "utils/GPUUtil.H"

void
DepositTemperature (PlasmaParticleContainer& plasma,
                    Fields & fields,
                    amrex::Vector<amrex::Geometry> const& gm,
                    int const lev)
{
}
