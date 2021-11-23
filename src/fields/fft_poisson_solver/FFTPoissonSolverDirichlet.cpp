#include "FFTPoissonSolverDirichlet.H"
#include "fft/AnyDST.H"
#include "utils/Constants.H"
#include "utils/HipaceProfilerWrapper.H"

FFTPoissonSolverDirichlet::FFTPoissonSolverDirichlet (
    amrex::BoxArray const& realspace_ba,
    amrex::DistributionMapping const& dm,
    amrex::Geometry const& gm )
{
    define(realspace_ba, dm, gm);
}

void
FFTPoissonSolverDirichlet::define (amrex::BoxArray const& a_realspace_ba,
                                   amrex::DistributionMapping const& dm,
                                   amrex::Geometry const& gm )
{
    using namespace amrex::literals;

    HIPACE_PROFILE("FFTPoissonSolverDirichlet::define()");
    // If we are going to support parallel FFT, the constructor needs to take a communicator.
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(a_realspace_ba.size() == 1, "Parallel FFT not supported yet");

    m_geom = gm;
    m_multipol_coeff.resize(33);
    amrex::ParmParse pp("hipace");
    queryWithParser(pp, "open_boundary", m_open_boundary);

    // Create the box array that corresponds to spectral space
    amrex::BoxList spectral_bl; // Create empty box list
    amrex::BoxList real_bl; // Create empty box list<
    // Loop over boxes and fill the box list
    for (int i=0; i < a_realspace_ba.size(); i++ ) {
        // For local FFTs, boxes in spectral space start at 0 in
        // each direction and have the same number of points as the
        // (cell-centered) real space box
        // Define the corresponding box
        amrex::Box real_box_ghost = a_realspace_ba[i];
        real_box_ghost.grow( Fields::m_slices_nguards );
        amrex::Box spectral_bx = amrex::Box( amrex::IntVect::TheZeroVector(),
                          real_box_ghost.length() - amrex::IntVect::TheUnitVector() );
        spectral_bl.push_back( spectral_bx );
        amrex::Box real_bx = spectral_bx;
        real_bx.setSmall(Direction::z, real_box_ghost.smallEnd(Direction::z));
        real_bx.setBig  (Direction::z, real_box_ghost.bigEnd(Direction::z));
        real_bl.push_back( real_bx );
    }
    m_spectralspace_ba.define( std::move(spectral_bl) );
    amrex::BoxArray real_ba(std::move(real_bl));

    // Allocate temporary arrays - in real space and spectral space
    // These arrays will store the data just before/after the FFT
    // The stagingArea is also created from 0 to nx, because the real space array may have
    // an offset for levels > 0
    m_stagingArea = amrex::MultiFab(real_ba, dm, 1, 0);
    m_tmpSpectralField = amrex::MultiFab(m_spectralspace_ba, dm, 1, 0);
    m_stagingArea.setVal(0.0); // this is not required
    m_tmpSpectralField.setVal(0.0);

    // This must be true even for parallel FFT.
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(m_stagingArea.local_size() == 1,
                                     "There should be only one box locally.");
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(m_tmpSpectralField.local_size() == 1,
                                     "There should be only one box locally.");

    const auto dx = gm.CellSizeArray();
    const amrex::Real dxsquared = dx[0]*dx[0];
    const amrex::Real dysquared = dx[1]*dx[1];
    const amrex::Real sine_x_factor = MathConst::pi / ( 2. * ( real_ba[0].length(0) + 1 ));
    const amrex::Real sine_y_factor = MathConst::pi / ( 2. * ( real_ba[0].length(1) + 1 ));

    // Normalization of FFTW's 'DST-I' discrete sine transform (FFTW_RODFT00)
    // This normalization is used regardless of the sine transform library
    const amrex::Real norm_fac = 0.5 / ( 2 * (( real_ba[0].length(0) + 1 )
                                             *( real_ba[0].length(1) + 1 )));

    m_eigenvalue_matrix = amrex::MultiFab(m_spectralspace_ba, dm, 1, 0);

    // Calculate the array of m_eigenvalue_matrix
    for (amrex::MFIter mfi(m_eigenvalue_matrix); mfi.isValid(); ++mfi ){
        amrex::Array4<amrex::Real> eigenvalue_matrix = m_eigenvalue_matrix.array(mfi);
        amrex::Box const& bx = mfi.validbox();  // The lower corner of the "2D" slice Box is zero.
        amrex::ParallelFor(
            bx, [=] AMREX_GPU_DEVICE (int i, int j, int /* k */) noexcept
                {
                    /* fast poisson solver diagonal x coeffs */
                    amrex::Real sinex_sq = sin(( i + 1 ) * sine_x_factor) * sin(( i + 1 ) * sine_x_factor);
                    /* fast poisson solver diagonal y coeffs */
                    amrex::Real siney_sq = sin(( j + 1 ) * sine_y_factor) * sin(( j + 1 ) * sine_y_factor);

                    if ((sinex_sq!=0) && (siney_sq!=0)) {
                        eigenvalue_matrix(i,j,0) = norm_fac / ( -4.0 * ( sinex_sq / dxsquared + siney_sq / dysquared ));
                    } else {
                        // Avoid division by 0
                        eigenvalue_matrix(i,j,0) = 0._rt;
                    }
                });
    }

    // Allocate and initialize the FFT plans
    m_plan = AnyDST::DSTplans(m_spectralspace_ba, dm);
    // Loop over boxes and allocate the corresponding plan
    // for each box owned by the local MPI proc
    for ( amrex::MFIter mfi(m_stagingArea); mfi.isValid(); ++mfi ){
        // Note: the size of the real-space box and spectral-space box
        // differ when using real-to-complex FFT. When initializing
        // the FFT plan, the valid dimensions are those of the real-space box.
        amrex::IntVect fft_size = mfi.validbox().length();
        m_plan[mfi] = AnyDST::CreatePlan(
            fft_size, &m_stagingArea[mfi], &m_tmpSpectralField[mfi]);
    }
}


void
FFTPoissonSolverDirichlet::AddBoundaryConditions (amrex::MFIter& mfi)
{
    HIPACE_PROFILE("FFTPoissonSolverDirichlet::AddBoundaryConditions()");
    using namespace amrex::literals;

    amrex::Array4<amrex::Real> rhs_array = m_stagingArea.array(mfi);
    const amrex::Box rhs_box = m_stagingArea[mfi].box();
    amrex::Real * multipol_coeff = m_multipol_coeff.data();
    const amrex::Real pi = MathConst::pi;

    for (amrex::Real& num : m_multipol_coeff) {
        num = 0.0_rt;
    }

    const int box_len0 = rhs_box.length(0);
    const int box_len1 = rhs_box.length(1);

    const int box_lo2 = rhs_box.smallEnd(2);

    const amrex::Real dx = m_geom.CellSize(0);
    const amrex::Real dy = m_geom.CellSize(1);

    {
    HIPACE_PROFILE("FFTPoissonSolverDirichlet::AddBoundaryConditions::Integral");
    amrex::ParallelFor( rhs_box,
        [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
            const amrex::Real scale = std::sqrt(box_len0*dx*box_len0*dy + box_len1*dy*box_len1*dy);
            const amrex::Real x = ((i + 0.5_rt - 0.5_rt*box_len0) * dx)/scale;
            const amrex::Real y = ((j + 0.5_rt - 0.5_rt*box_len1) * dy)/scale;

            if( (i==0)||(j==0)||(i==box_len0-1)||(j==box_len1-1) ) {
                rhs_array(i, j, k) = 0;
            }

            const amrex::Real source_val = rhs_array(i, j, k);

            amrex::Gpu::Atomic::AddNoRet( multipol_coeff + 0, source_val );
            amrex::Gpu::Atomic::AddNoRet( multipol_coeff + 1, source_val*x );
            amrex::Gpu::Atomic::AddNoRet( multipol_coeff + 2, source_val*y );
            amrex::Gpu::Atomic::AddNoRet( multipol_coeff + 3, source_val*(-x*x + y*y) );
            amrex::Gpu::Atomic::AddNoRet( multipol_coeff + 4, source_val*x*y );
            amrex::Gpu::Atomic::AddNoRet( multipol_coeff + 5, source_val*(x*x*x - 3*x*y*y) );
            amrex::Gpu::Atomic::AddNoRet( multipol_coeff + 6, source_val*(-3*x*x*y + y*y*y) );
            amrex::Gpu::Atomic::AddNoRet( multipol_coeff + 7, source_val*(x*x*x*x - 6*x*x*y*y + y*y*y*y) );
            amrex::Gpu::Atomic::AddNoRet( multipol_coeff + 8, source_val*(-x*x*x*y + x*y*y*y) );
            amrex::Gpu::Atomic::AddNoRet( multipol_coeff + 9, source_val*(x*x*x*x*x - 10*x*x*x*y*y + 5*x*y*y*y*y) );
            amrex::Gpu::Atomic::AddNoRet( multipol_coeff + 10, source_val*(5*x*x*x*x*y - 10*x*x*y*y*y + y*y*y*y*y) );
            amrex::Gpu::Atomic::AddNoRet( multipol_coeff + 11, source_val*(-x*x*x*x*x*x + 15*x*x*x*x*y*y - 15*x*x*y*y*y*y + y*y*y*y*y*y) );
            amrex::Gpu::Atomic::AddNoRet( multipol_coeff + 12, source_val*(x*x*x*x*x*y - 10.0_rt/3.0_rt*x*x*x*y*y*y + x*y*y*y*y*y) );
            amrex::Gpu::Atomic::AddNoRet( multipol_coeff + 13, source_val*(x*x*x*x*x*x*x - 21*x*x*x*x*x*y*y + 35*x*x*x*y*y*y*y - 7*x*y*y*y*y*y*y) );
            amrex::Gpu::Atomic::AddNoRet( multipol_coeff + 14, source_val*(-7*x*x*x*x*x*x*y + 35*x*x*x*x*y*y*y - 21*x*x*y*y*y*y*y + y*y*y*y*y*y*y) );
            amrex::Gpu::Atomic::AddNoRet( multipol_coeff + 15, source_val*(x*x*x*x*x*x*x*x - 28*x*x*x*x*x*x*y*y + 70*x*x*x*x*y*y*y*y - 28*x*x*y*y*y*y*y*y + y*y*y*y*y*y*y*y) );
            amrex::Gpu::Atomic::AddNoRet( multipol_coeff + 16, source_val*(-x*x*x*x*x*x*x*y + 7*x*x*x*x*x*y*y*y - 7*x*x*x*y*y*y*y*y + x*y*y*y*y*y*y*y) );
            amrex::Gpu::Atomic::AddNoRet( multipol_coeff + 17, source_val*(x*x*x*x*x*x*x*x*x - 36*x*x*x*x*x*x*x*y*y + 126*x*x*x*x*x*y*y*y*y - 84*x*x*x*y*y*y*y*y*y + 9*x*y*y*y*y*y*y*y*y) );
            amrex::Gpu::Atomic::AddNoRet( multipol_coeff + 18, source_val*(9*x*x*x*x*x*x*x*x*y - 84*x*x*x*x*x*x*y*y*y + 126*x*x*x*x*y*y*y*y*y - 36*x*x*y*y*y*y*y*y*y + y*y*y*y*y*y*y*y*y) );
            amrex::Gpu::Atomic::AddNoRet( multipol_coeff + 19, source_val*(-x*x*x*x*x*x*x*x*x*x + 45*x*x*x*x*x*x*x*x*y*y - 210*x*x*x*x*x*x*y*y*y*y + 210*x*x*x*x*y*y*y*y*y*y - 45*x*x*y*y*y*y*y*y*y*y + y*y*y*y*y*y*y*y*y*y) );
            amrex::Gpu::Atomic::AddNoRet( multipol_coeff + 20, source_val*(x*x*x*x*x*x*x*x*x*y - 12*x*x*x*x*x*x*x*y*y*y + (126.0_rt/5.0_rt)*x*x*x*x*x*y*y*y*y*y - 12*x*x*x*y*y*y*y*y*y*y + x*y*y*y*y*y*y*y*y*y) );
            amrex::Gpu::Atomic::AddNoRet( multipol_coeff + 21, source_val*(x*x*x*x*x*x*x*x*x*x*x - 55*x*x*x*x*x*x*x*x*x*y*y + 330*x*x*x*x*x*x*x*y*y*y*y - 462*x*x*x*x*x*y*y*y*y*y*y + 165*x*x*x*y*y*y*y*y*y*y*y - 11*x*y*y*y*y*y*y*y*y*y*y) );
            amrex::Gpu::Atomic::AddNoRet( multipol_coeff + 22, source_val*(-11*x*x*x*x*x*x*x*x*x*x*y + 165*x*x*x*x*x*x*x*x*y*y*y - 462*x*x*x*x*x*x*y*y*y*y*y + 330*x*x*x*x*y*y*y*y*y*y*y - 55*x*x*y*y*y*y*y*y*y*y*y + y*y*y*y*y*y*y*y*y*y*y) );
            amrex::Gpu::Atomic::AddNoRet( multipol_coeff + 23, source_val*(x*x*x*x*x*x*x*x*x*x*x*x - 66*x*x*x*x*x*x*x*x*x*x*y*y + 495*x*x*x*x*x*x*x*x*y*y*y*y - 924*x*x*x*x*x*x*y*y*y*y*y*y + 495*x*x*x*x*y*y*y*y*y*y*y*y - 66*x*x*y*y*y*y*y*y*y*y*y*y + y*y*y*y*y*y*y*y*y*y*y*y) );
            amrex::Gpu::Atomic::AddNoRet( multipol_coeff + 24, source_val*(-x*x*x*x*x*x*x*x*x*x*x*y + (55.0_rt/3.0_rt)*x*x*x*x*x*x*x*x*x*y*y*y - 66*x*x*x*x*x*x*x*y*y*y*y*y + 66*x*x*x*x*x*y*y*y*y*y*y*y - 55.0_rt/3.0_rt*x*x*x*y*y*y*y*y*y*y*y*y + x*y*y*y*y*y*y*y*y*y*y*y) );
            amrex::Gpu::Atomic::AddNoRet( multipol_coeff + 25, source_val*(x*x*x*x*x*x*x*x*x*x*x*x*x - 78*x*x*x*x*x*x*x*x*x*x*x*y*y + 715*x*x*x*x*x*x*x*x*x*y*y*y*y - 1716*x*x*x*x*x*x*x*y*y*y*y*y*y + 1287*x*x*x*x*x*y*y*y*y*y*y*y*y - 286*x*x*x*y*y*y*y*y*y*y*y*y*y + 13*x*y*y*y*y*y*y*y*y*y*y*y*y) );
            amrex::Gpu::Atomic::AddNoRet( multipol_coeff + 26, source_val*(13*x*x*x*x*x*x*x*x*x*x*x*x*y - 286*x*x*x*x*x*x*x*x*x*x*y*y*y + 1287*x*x*x*x*x*x*x*x*y*y*y*y*y - 1716*x*x*x*x*x*x*y*y*y*y*y*y*y + 715*x*x*x*x*y*y*y*y*y*y*y*y*y - 78*x*x*y*y*y*y*y*y*y*y*y*y*y + y*y*y*y*y*y*y*y*y*y*y*y*y) );
            amrex::Gpu::Atomic::AddNoRet( multipol_coeff + 27, source_val*(-x*x*x*x*x*x*x*x*x*x*x*x*x*x + 91*x*x*x*x*x*x*x*x*x*x*x*x*y*y - 1001*x*x*x*x*x*x*x*x*x*x*y*y*y*y + 3003*x*x*x*x*x*x*x*x*y*y*y*y*y*y - 3003*x*x*x*x*x*x*y*y*y*y*y*y*y*y + 1001*x*x*x*x*y*y*y*y*y*y*y*y*y*y - 91*x*x*y*y*y*y*y*y*y*y*y*y*y*y + y*y*y*y*y*y*y*y*y*y*y*y*y*y) );
            amrex::Gpu::Atomic::AddNoRet( multipol_coeff + 28, source_val*(x*x*x*x*x*x*x*x*x*x*x*x*x*y - 26*x*x*x*x*x*x*x*x*x*x*x*y*y*y + 143*x*x*x*x*x*x*x*x*x*y*y*y*y*y - 1716.0_rt/7.0_rt*x*x*x*x*x*x*x*y*y*y*y*y*y*y + 143*x*x*x*x*x*y*y*y*y*y*y*y*y*y - 26*x*x*x*y*y*y*y*y*y*y*y*y*y*y + x*y*y*y*y*y*y*y*y*y*y*y*y*y) );
            amrex::Gpu::Atomic::AddNoRet( multipol_coeff + 29, source_val*(x*x*x*x*x*x*x*x*x*x*x*x*x*x*x - 105*x*x*x*x*x*x*x*x*x*x*x*x*x*y*y + 1365*x*x*x*x*x*x*x*x*x*x*x*y*y*y*y - 5005*x*x*x*x*x*x*x*x*x*y*y*y*y*y*y + 6435*x*x*x*x*x*x*x*y*y*y*y*y*y*y*y - 3003*x*x*x*x*x*y*y*y*y*y*y*y*y*y*y + 455*x*x*x*y*y*y*y*y*y*y*y*y*y*y*y - 15*x*y*y*y*y*y*y*y*y*y*y*y*y*y*y) );
            amrex::Gpu::Atomic::AddNoRet( multipol_coeff + 30, source_val*(-15*x*x*x*x*x*x*x*x*x*x*x*x*x*x*y + 455*x*x*x*x*x*x*x*x*x*x*x*x*y*y*y - 3003*x*x*x*x*x*x*x*x*x*x*y*y*y*y*y + 6435*x*x*x*x*x*x*x*x*y*y*y*y*y*y*y - 5005*x*x*x*x*x*x*y*y*y*y*y*y*y*y*y + 1365*x*x*x*x*y*y*y*y*y*y*y*y*y*y*y - 105*x*x*y*y*y*y*y*y*y*y*y*y*y*y*y + y*y*y*y*y*y*y*y*y*y*y*y*y*y*y) );
            amrex::Gpu::Atomic::AddNoRet( multipol_coeff + 31, source_val*(x*x*x*x*x*x*x*x*x*x*x*x*x*x*x*x - 120*x*x*x*x*x*x*x*x*x*x*x*x*x*x*y*y + 1820*x*x*x*x*x*x*x*x*x*x*x*x*y*y*y*y - 8008*x*x*x*x*x*x*x*x*x*x*y*y*y*y*y*y + 12870*x*x*x*x*x*x*x*x*y*y*y*y*y*y*y*y - 8008*x*x*x*x*x*x*y*y*y*y*y*y*y*y*y*y + 1820*x*x*x*x*y*y*y*y*y*y*y*y*y*y*y*y - 120*x*x*y*y*y*y*y*y*y*y*y*y*y*y*y*y + y*y*y*y*y*y*y*y*y*y*y*y*y*y*y*y) );
            amrex::Gpu::Atomic::AddNoRet( multipol_coeff + 32, source_val*(-x*x*x*x*x*x*x*x*x*x*x*x*x*x*x*y + 35*x*x*x*x*x*x*x*x*x*x*x*x*x*y*y*y - 273*x*x*x*x*x*x*x*x*x*x*x*y*y*y*y*y + 715*x*x*x*x*x*x*x*x*x*y*y*y*y*y*y*y - 715*x*x*x*x*x*x*x*y*y*y*y*y*y*y*y*y + 273*x*x*x*x*x*y*y*y*y*y*y*y*y*y*y*y - 35*x*x*x*y*y*y*y*y*y*y*y*y*y*y*y*y + x*y*y*y*y*y*y*y*y*y*y*y*y*y*y*y) );
        });
    }

    std::cout << "field:\n[";
    //for (amrex::Real& num : m_multipol_coeff) {
    //    std::cout << " " << num;
    //}
    for(int i=0; i<box_len0; ++i) {
        std::cout << "[  ";
        for(int j=0; j<box_len1; ++j) {
            std::cout << rhs_array(i, j, box_lo2) << ", ";
        }
        std::cout << " ],\n";
    }
    std::cout << "]\nmultipol coeffs:\n[";

    for (amrex::Real& num : m_multipol_coeff) {
        std::cout << " " << num << ",";
    }
    std::cout << "]\n" << std::endl;

    {
    HIPACE_PROFILE("FFTPoissonSolverDirichlet::AddBoundaryConditions::Evaluate");
    amrex::ParallelFor( {{0, 0, 0}, {box_len0 + box_len1 - 1, 1, 0}},
        [=] AMREX_GPU_DEVICE(int ie, int je, int) noexcept {
            const bool j_is_changing = (ie >= box_len0);
            const int edge = j_is_changing + 2*je;
            const int i = (!j_is_changing)*(ie) + (edge==1)*(-1) + (edge==3)*(box_len0);
            const int j = (edge==0)*(-1) + (j_is_changing)*(ie-box_len0) + (edge==2)*(box_len1);

            const amrex::Real scale = std::sqrt(box_len0*dx*box_len0*dy + box_len1*dy*box_len1*dy);
            const amrex::Real x_s = ((i + 0.5_rt - 0.5_rt*box_len0) * dx)/scale;
            const amrex::Real y_s = ((j + 0.5_rt - 0.5_rt*box_len1) * dy)/scale;
            const amrex::Real r2 = x_s*x_s + y_s*y_s;
            const amrex::Real x = x_s/r2;
            const amrex::Real y = y_s/r2;

            const amrex::Real val =
                multipol_coeff[0] * (std::log(6*r2))
              + multipol_coeff[1] * (-2*x)
              + multipol_coeff[2] * (-2*y)
              + multipol_coeff[3] * (x*x - y*y)
              + multipol_coeff[4] * (-4*x*y)
              + multipol_coeff[5] * (-2.0_rt/3.0_rt*x*x*x + 2*x*y*y)
              + multipol_coeff[6] * (2*x*x*y - 2.0_rt/3.0_rt*y*y*y)
              + multipol_coeff[7] * (-1.0_rt/2.0_rt*x*x*x*x + 3*x*x*y*y - 1.0_rt/2.0_rt*y*y*y*y)
              + multipol_coeff[8] * (8*x*x*x*y - 8*x*y*y*y)
              + multipol_coeff[9] * (-2.0_rt/5.0_rt*x*x*x*x*x + 4*x*x*x*y*y - 2*x*y*y*y*y)
              + multipol_coeff[10] * (-2*x*x*x*x*y + 4*x*x*y*y*y - 2.0_rt/5.0_rt*y*y*y*y*y)
              + multipol_coeff[11] * ((1.0_rt/3.0_rt)*x*x*x*x*x*x - 5*x*x*x*x*y*y + 5*x*x*y*y*y*y - 1.0_rt/3.0_rt*y*y*y*y*y*y)
              + multipol_coeff[12] * (-12*x*x*x*x*x*y + 40*x*x*x*y*y*y - 12*x*y*y*y*y*y)
              + multipol_coeff[13] * (-2.0_rt/7.0_rt*x*x*x*x*x*x*x + 6*x*x*x*x*x*y*y - 10*x*x*x*y*y*y*y + 2*x*y*y*y*y*y*y)
              + multipol_coeff[14] * (2*x*x*x*x*x*x*y - 10*x*x*x*x*y*y*y + 6*x*x*y*y*y*y*y - 2.0_rt/7.0_rt*y*y*y*y*y*y*y)
              + multipol_coeff[15] * (-1.0_rt/4.0_rt*x*x*x*x*x*x*x*x + 7*x*x*x*x*x*x*y*y - 35.0_rt/2.0_rt*x*x*x*x*y*y*y*y + 7*x*x*y*y*y*y*y*y - 1.0_rt/4.0_rt*y*y*y*y*y*y*y*y)
              + multipol_coeff[16] * (16*x*x*x*x*x*x*x*y - 112*x*x*x*x*x*y*y*y + 112*x*x*x*y*y*y*y*y - 16*x*y*y*y*y*y*y*y)
              + multipol_coeff[17] * (-2.0_rt/9.0_rt*x*x*x*x*x*x*x*x*x + 8*x*x*x*x*x*x*x*y*y - 28*x*x*x*x*x*y*y*y*y + (56.0_rt/3.0_rt)*x*x*x*y*y*y*y*y*y - 2*x*y*y*y*y*y*y*y*y)
              + multipol_coeff[18] * (-2*x*x*x*x*x*x*x*x*y + (56.0_rt/3.0_rt)*x*x*x*x*x*x*y*y*y - 28*x*x*x*x*y*y*y*y*y + 8*x*x*y*y*y*y*y*y*y - 2.0_rt/9.0_rt*y*y*y*y*y*y*y*y*y)
              + multipol_coeff[19] * ((1.0_rt/5.0_rt)*x*x*x*x*x*x*x*x*x*x - 9*x*x*x*x*x*x*x*x*y*y + 42*x*x*x*x*x*x*y*y*y*y - 42*x*x*x*x*y*y*y*y*y*y + 9*x*x*y*y*y*y*y*y*y*y - 1.0_rt/5.0_rt*y*y*y*y*y*y*y*y*y*y)
              + multipol_coeff[20] * (-20*x*x*x*x*x*x*x*x*x*y + 240*x*x*x*x*x*x*x*y*y*y - 504*x*x*x*x*x*y*y*y*y*y + 240*x*x*x*y*y*y*y*y*y*y - 20*x*y*y*y*y*y*y*y*y*y)
              + multipol_coeff[21] * (-2.0_rt/11.0_rt*x*x*x*x*x*x*x*x*x*x*x + 10*x*x*x*x*x*x*x*x*x*y*y - 60*x*x*x*x*x*x*x*y*y*y*y + 84*x*x*x*x*x*y*y*y*y*y*y - 30*x*x*x*y*y*y*y*y*y*y*y + 2*x*y*y*y*y*y*y*y*y*y*y)
              + multipol_coeff[22] * (2*x*x*x*x*x*x*x*x*x*x*y - 30*x*x*x*x*x*x*x*x*y*y*y + 84*x*x*x*x*x*x*y*y*y*y*y - 60*x*x*x*x*y*y*y*y*y*y*y + 10*x*x*y*y*y*y*y*y*y*y*y - 2.0_rt/11.0_rt*y*y*y*y*y*y*y*y*y*y*y)
              + multipol_coeff[23] * (-1.0_rt/6.0_rt*x*x*x*x*x*x*x*x*x*x*x*x + 11*x*x*x*x*x*x*x*x*x*x*y*y - 165.0_rt/2.0_rt*x*x*x*x*x*x*x*x*y*y*y*y + 154*x*x*x*x*x*x*y*y*y*y*y*y - 165.0_rt/2.0_rt*x*x*x*x*y*y*y*y*y*y*y*y + 11*x*x*y*y*y*y*y*y*y*y*y*y - 1.0_rt/6.0_rt*y*y*y*y*y*y*y*y*y*y*y*y)
              + multipol_coeff[24] * (24*x*x*x*x*x*x*x*x*x*x*x*y - 440*x*x*x*x*x*x*x*x*x*y*y*y + 1584*x*x*x*x*x*x*x*y*y*y*y*y - 1584*x*x*x*x*x*y*y*y*y*y*y*y + 440*x*x*x*y*y*y*y*y*y*y*y*y - 24*x*y*y*y*y*y*y*y*y*y*y*y)
              + multipol_coeff[25] * (-2.0_rt/13.0_rt*x*x*x*x*x*x*x*x*x*x*x*x*x + 12*x*x*x*x*x*x*x*x*x*x*x*y*y - 110*x*x*x*x*x*x*x*x*x*y*y*y*y + 264*x*x*x*x*x*x*x*y*y*y*y*y*y - 198*x*x*x*x*x*y*y*y*y*y*y*y*y + 44*x*x*x*y*y*y*y*y*y*y*y*y*y - 2*x*y*y*y*y*y*y*y*y*y*y*y*y)
              + multipol_coeff[26] * (-2*x*x*x*x*x*x*x*x*x*x*x*x*y + 44*x*x*x*x*x*x*x*x*x*x*y*y*y - 198*x*x*x*x*x*x*x*x*y*y*y*y*y + 264*x*x*x*x*x*x*y*y*y*y*y*y*y - 110*x*x*x*x*y*y*y*y*y*y*y*y*y + 12*x*x*y*y*y*y*y*y*y*y*y*y*y - 2.0_rt/13.0_rt*y*y*y*y*y*y*y*y*y*y*y*y*y)
              + multipol_coeff[27] * ((1.0_rt/7.0_rt)*x*x*x*x*x*x*x*x*x*x*x*x*x*x - 13*x*x*x*x*x*x*x*x*x*x*x*x*y*y + 143*x*x*x*x*x*x*x*x*x*x*y*y*y*y - 429*x*x*x*x*x*x*x*x*y*y*y*y*y*y + 429*x*x*x*x*x*x*y*y*y*y*y*y*y*y - 143*x*x*x*x*y*y*y*y*y*y*y*y*y*y + 13*x*x*y*y*y*y*y*y*y*y*y*y*y*y - 1.0_rt/7.0_rt*y*y*y*y*y*y*y*y*y*y*y*y*y*y)
              + multipol_coeff[28] * (-28*x*x*x*x*x*x*x*x*x*x*x*x*x*y + 728*x*x*x*x*x*x*x*x*x*x*x*y*y*y - 4004*x*x*x*x*x*x*x*x*x*y*y*y*y*y + 6864*x*x*x*x*x*x*x*y*y*y*y*y*y*y - 4004*x*x*x*x*x*y*y*y*y*y*y*y*y*y + 728*x*x*x*y*y*y*y*y*y*y*y*y*y*y - 28*x*y*y*y*y*y*y*y*y*y*y*y*y*y)
              + multipol_coeff[29] * (-2.0_rt/15.0_rt*x*x*x*x*x*x*x*x*x*x*x*x*x*x*x + 14*x*x*x*x*x*x*x*x*x*x*x*x*x*y*y - 182*x*x*x*x*x*x*x*x*x*x*x*y*y*y*y + (2002.0_rt/3.0_rt)*x*x*x*x*x*x*x*x*x*y*y*y*y*y*y - 858*x*x*x*x*x*x*x*y*y*y*y*y*y*y*y + (2002.0_rt/5.0_rt)*x*x*x*x*x*y*y*y*y*y*y*y*y*y*y - 182.0_rt/3.0_rt*x*x*x*y*y*y*y*y*y*y*y*y*y*y*y + 2*x*y*y*y*y*y*y*y*y*y*y*y*y*y*y)
              + multipol_coeff[30] * (2*x*x*x*x*x*x*x*x*x*x*x*x*x*x*y - 182.0_rt/3.0_rt*x*x*x*x*x*x*x*x*x*x*x*x*y*y*y + (2002.0_rt/5.0_rt)*x*x*x*x*x*x*x*x*x*x*y*y*y*y*y - 858*x*x*x*x*x*x*x*x*y*y*y*y*y*y*y + (2002.0_rt/3.0_rt)*x*x*x*x*x*x*y*y*y*y*y*y*y*y*y - 182*x*x*x*x*y*y*y*y*y*y*y*y*y*y*y + 14*x*x*y*y*y*y*y*y*y*y*y*y*y*y*y - 2.0_rt/15.0_rt*y*y*y*y*y*y*y*y*y*y*y*y*y*y*y)
              + multipol_coeff[31] * (-1.0_rt/8.0_rt*x*x*x*x*x*x*x*x*x*x*x*x*x*x*x*x + 15*x*x*x*x*x*x*x*x*x*x*x*x*x*x*y*y - 455.0_rt/2.0_rt*x*x*x*x*x*x*x*x*x*x*x*x*y*y*y*y + 1001*x*x*x*x*x*x*x*x*x*x*y*y*y*y*y*y - 6435.0_rt/4.0_rt*x*x*x*x*x*x*x*x*y*y*y*y*y*y*y*y + 1001*x*x*x*x*x*x*y*y*y*y*y*y*y*y*y*y - 455.0_rt/2.0_rt*x*x*x*x*y*y*y*y*y*y*y*y*y*y*y*y + 15*x*x*y*y*y*y*y*y*y*y*y*y*y*y*y*y - 1.0_rt/8.0_rt*y*y*y*y*y*y*y*y*y*y*y*y*y*y*y*y)
              + multipol_coeff[32] * (32*x*x*x*x*x*x*x*x*x*x*x*x*x*x*x*y - 1120*x*x*x*x*x*x*x*x*x*x*x*x*x*y*y*y + 8736*x*x*x*x*x*x*x*x*x*x*x*y*y*y*y*y - 22880*x*x*x*x*x*x*x*x*x*y*y*y*y*y*y*y + 22880*x*x*x*x*x*x*x*y*y*y*y*y*y*y*y*y - 8736*x*x*x*x*x*y*y*y*y*y*y*y*y*y*y*y + 1120*x*x*x*y*y*y*y*y*y*y*y*y*y*y*y*y - 32*x*y*y*y*y*y*y*y*y*y*y*y*y*y*y*y)

            ;amrex::Gpu::Atomic::AddNoRet( &(rhs_array(i+(edge==1)-(edge==3), j+(edge==0)-(edge==2), box_lo2)), -val/(4*pi) );
        });
    }

    std::cout << "field with boundary:\n[";
    for(int i=0; i<box_len0; ++i) {
        std::cout << "[  ";
        for(int j=0; j<box_len1; ++j) {
            std::cout << rhs_array(i, j, box_lo2) << ", ";
        }
        std::cout << " ],\n";
    }
    std::cout << "]\n" << std::endl;
}


void
FFTPoissonSolverDirichlet::SolvePoissonEquation (amrex::MultiFab& lhs_mf)
{
    HIPACE_PROFILE("FFTPoissonSolverDirichlet::SolvePoissonEquation()");

    // Loop over boxes
    for ( amrex::MFIter mfi(m_stagingArea); mfi.isValid(); ++mfi ){

        if (m_open_boundary) AddBoundaryConditions(mfi);

        // Perform Fourier transform from the staging area to `tmpSpectralField`
        AnyDST::Execute<AnyDST::direction::forward>(m_plan[mfi]);

        // Solve Poisson equation in Fourier space:
        // Multiply `tmpSpectralField` by eigenvalue_matrix
        amrex::Array4<amrex::Real> tmp_cmplx_arr = m_tmpSpectralField.array(mfi);
        amrex::Array4<amrex::Real> eigenvalue_matrix = m_eigenvalue_matrix.array(mfi);

        amrex::ParallelFor( m_spectralspace_ba[mfi],
            [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
                tmp_cmplx_arr(i,j,k) *= eigenvalue_matrix(i,j,k);
            });

        // Perform Fourier transform from `tmpSpectralField` to the staging area
        AnyDST::Execute<AnyDST::direction::backward>(m_plan[mfi]);

        // Copy from the staging area to output array (and normalize)
        amrex::Array4<amrex::Real> tmp_real_arr = m_stagingArea.array(mfi);
        amrex::Array4<amrex::Real> lhs_arr = lhs_mf.array(mfi);
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(lhs_mf.size() == 1,
                                         "Slice MFs must be defined on one box only");
        const amrex::FArrayBox& lhs_fab = lhs_mf[0];
        const amrex::IntVect lo = lhs_fab.box().smallEnd();
        amrex::ParallelFor( mfi.validbox(),
            [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
                // Copy and normalize field
                lhs_arr(i+lo[0],j+lo[1],k) = tmp_real_arr(i,j,k);
            });
    }
}
