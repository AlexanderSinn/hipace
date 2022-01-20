#include "Diagnostic.H"
#include "Hipace.H"
#include <AMReX_ParmParse.H>

Diagnostic::Diagnostic (int nlev)
    : m_F(nlev),
      m_diag_coarsen(nlev),
      m_geom_io(nlev)
{
    amrex::ParmParse ppd("diagnostic");
    std::string str_type;
    getWithParser(ppd, "diag_type", str_type);
    if        (str_type == "xyz"){
        m_diag_type = DiagType::xyz;
        m_slice_dir = -1;
    } else if (str_type == "xz") {
        m_diag_type = DiagType::xz;
        m_slice_dir = 1;
    } else if (str_type == "yz") {
        m_diag_type = DiagType::yz;
        m_slice_dir = 0;
    } else {
        amrex::Abort("Unknown diagnostics type: must be xyz, xz or yz.");
    }

    for(int ilev = 0; ilev<nlev; ++ilev) {
        amrex::Array<int,3> diag_coarsen_arr{1,1,1};
        // set all levels the same for now
        queryWithParser(ppd, "coarsening", diag_coarsen_arr);
        if(m_slice_dir == 0 || m_slice_dir == 1) {
            diag_coarsen_arr[m_slice_dir] = 1;
        }
        m_diag_coarsen[ilev] = amrex::IntVect(diag_coarsen_arr);
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE( m_diag_coarsen[ilev].min() >= 1,
            "Coarsening ratio must be >= 1");
    }

    queryWithParser(ppd, "field_data", m_comps_output);
    const amrex::Vector<std::string> all_field_comps
            {"ExmBy", "EypBx", "Ez", "Bx", "By", "Bz", "jx", "jx_beam", "jy", "jy_beam", "jz",
             "jz_beam", "rho", "Psi"};
    if(m_comps_output.empty()) {
        m_comps_output = all_field_comps;
    }
    else {
        for(std::string comp_name : m_comps_output) {
            if(comp_name == "all" || comp_name == "All") {
                m_comps_output = all_field_comps;
                break;
            }
            if(comp_name == "none" || comp_name == "None") {
                m_comps_output.clear();
                break;
            }
            if(Comps[WhichSlice::This].count(comp_name) == 0 || comp_name == "N") {
                amrex::Abort("Unknown field diagnostics component: " + comp_name + "\nmust be " +
                "'all', 'none' or a subset of: ExmBy EypBx Ez Bx By Bz jx jy jz jx_beam jy_beam " +
                "jz_beam rho Psi" );
            }
        }
    }
    m_nfields = m_comps_output.size();
    m_comps_output_idx = amrex::Gpu::DeviceVector<int>(m_nfields);
    for(int i = 0; i < m_nfields; ++i) {
        m_comps_output_idx[i] = Comps[WhichSlice::This][m_comps_output[i]];
    }

    amrex::ParmParse ppb("beams");
    // read in all beam names
    amrex::Vector<std::string> all_beam_names;
    queryWithParser(ppb, "names", all_beam_names);
    // read in which beam should be written to file
    queryWithParser(ppd, "beam_data", m_output_beam_names);

    if(m_output_beam_names.empty()) {
        m_output_beam_names = all_beam_names;
    } else {
        for(std::string beam_name : m_output_beam_names) {
            if(beam_name == "all" || beam_name == "All") {
                m_output_beam_names = all_beam_names;
                break;
            }
            if(beam_name == "none" || beam_name == "None") {
                m_output_beam_names.clear();
                break;
            }
            if(std::find(all_beam_names.begin(), all_beam_names.end(), beam_name) ==  all_beam_names.end() ) {
                amrex::Abort("Unknown beam name: " + beam_name + "\nmust be " +
                "a subset of beams.names or 'none'");
            }
        }
    }
}

void
Diagnostic::AllocData (int lev, const amrex::Box& bx, amrex::Geometry const& geom)
{
    // trim the 3D box to slice box for slice IO
    amrex::Box F_bx = TrimIOBox(bx);

    F_bx.coarsen(m_diag_coarsen[lev]);

    m_F.push_back(amrex::FArrayBox(F_bx, m_nfields, amrex::The_Pinned_Arena()));

    m_geom_io[lev] = geom;
    amrex::RealBox prob_domain = geom.ProbDomain();
    amrex::Box domain = geom.Domain();
    // Define slice box
    if (m_slice_dir >= 0){
        int const icenter = domain.length(m_slice_dir)/2;
        domain.setSmall(m_slice_dir, icenter);
        domain.setBig(m_slice_dir, icenter);
        m_geom_io[lev] = amrex::Geometry(domain, &prob_domain, geom.Coord());
    }
    m_geom_io[lev].coarsen(m_diag_coarsen[lev]);
}

void
Diagnostic::ResizeFDiagFAB (const amrex::Box box, const int lev)
{
    amrex::Box io_box = TrimIOBox(box);
    io_box.coarsen(m_diag_coarsen[lev]);
    m_F[lev].resize(io_box, m_nfields);
    m_F[lev].setVal<amrex::RunOn::Device>(0);
}

amrex::Box
Diagnostic::TrimIOBox (const amrex::Box box_3d)
{
    // Create a xz slice Box
    amrex::Box slice_bx = box_3d;
    if (m_slice_dir >= 0){
            // Flatten the box down to 1 cell in the approprate direction.
            const int idx = box_3d.smallEnd(m_slice_dir) + box_3d.length(m_slice_dir)/2;
            slice_bx.setSmall(m_slice_dir, idx);
            slice_bx.setBig  (m_slice_dir, idx);
    }
    // m_F is defined on F_bx, the full or the slice Box
    amrex::Box F_bx = m_slice_dir >= 0 ? slice_bx : box_3d;

    return F_bx;
}
