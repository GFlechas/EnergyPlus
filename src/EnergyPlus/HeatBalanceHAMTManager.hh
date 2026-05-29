// EnergyPlus, Copyright (c) 1996-present, The Board of Trustees of the University of Illinois,
// The Regents of the University of California, through Lawrence Berkeley National Laboratory
// (subject to receipt of any required approvals from the U.S. Dept. of Energy), Oak Ridge
// National Laboratory, managed by UT-Battelle, Alliance for Energy Innovation, LLC, and other
// contributors. All rights reserved.
//
// NOTICE: This Software was developed under funding from the U.S. Department of Energy and the
// U.S. Government consequently retains certain rights. As such, the U.S. Government has been
// granted for itself and others acting on its behalf a paid-up, nonexclusive, irrevocable,
// worldwide license in the Software to reproduce, distribute copies to the public, prepare
// derivative works, and perform publicly and display publicly, and to permit others to do so.
//
// Redistribution and use in source and binary forms, with or without modification, are permitted
// provided that the following conditions are met:
//
// (1) Redistributions of source code must retain the above copyright notice, this list of
//     conditions and the following disclaimer.
//
// (2) Redistributions in binary form must reproduce the above copyright notice, this list of
//     conditions and the following disclaimer in the documentation and/or other materials
//     provided with the distribution.
//
// (3) Neither the name of the University of California, Lawrence Berkeley National Laboratory,
//     the University of Illinois, U.S. Dept. of Energy nor the names of its contributors may be
//     used to endorse or promote products derived from this software without specific prior
//     written permission.
//
// (4) Use of EnergyPlus(TM) Name. If Licensee (i) distributes the software in stand-alone form
//     without changes from the version obtained under this License, or (ii) Licensee makes a
//     reference solely to the software portion of its product, Licensee must refer to the
//     software as "EnergyPlus version X" software, where "X" is the version number Licensee
//     obtained under this License and may not use a different name for the software. Except as
//     specifically required in this Section (4), Licensee shall not use in a company name, a
//     product name, in advertising, publicity, or other promotional activities any name, trade
//     name, trademark, logo, or other designation of "EnergyPlus", "E+", "e+" or confusingly
//     similar designation, without the U.S. Department of Energy's prior written consent.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR
// IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY
// AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
// SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
// THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
// OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

#ifndef HeatBalanceHAMTManager_hh_INCLUDED
#define HeatBalanceHAMTManager_hh_INCLUDED

// ObjexxFCL Headers
#include <ObjexxFCL/Optional.hh>

// EnergyPlus Headers
#include <EnergyPlus/Data/BaseData.hh>
#include <EnergyPlus/EnergyPlus.hh>
#include <EnergyPlus/Material.hh>

namespace EnergyPlus {

// Forward declarations
struct EnergyPlusData;

namespace HeatBalanceHAMTManager {

    // MODULE INFORMATION:
    //       AUTHOR         Phillip Biddulph (June 2008) - original Gauss-Seidel HAMT model
    //       MODIFIED       May 2026: Gabriel Flechas, PhD - added Thomas (backward-Euler) and
    //                      second-order BDF2 (Backward Differentiation Formula) direct-solver
    //                      schemes, Fourier-number meshing, and new output variables. Original
    //                      Gauss-Seidel scheme retained.
    //                      Developed with the assistance of Anthropic Claude models
    //                      (Sonnet 4.6, Opus 4.7, Opus 4.8). See HeatBalanceHAMTManager.cc for the
    //                      full module header, methodology, and references.

    enum class HAMTScheme
    {
        GaussSeidel,               // legacy iterative point solver (June 2008); selectable, unchanged
        FullyImplicitThomas,       // TDMA direct solver, backward-Euler in time (1st order)
        FullyImplicitSecondOrder   // TDMA direct solver, BDF2 (second-order Backward Differentiation
                                   // Formula) in time (2nd order) - default scheme
    };

    // Data
    // MODULE PARAMETER DEFINITIONS:
    //
    // Uniform-grid resample of an HAMT property table. Built once from the
    // raw isorh/isodata arrays after IDF input, then used in the Picard inner
    // loop for O(1) lookup instead of the O(ndata) linear search in interp().
    //
    // Indexing: given x in [x0, x0 + (N-1)*dx], compute
    //     fidx = (x - x0) * inv_dx        // float index
    //     idx  = clamp((int)fidx, 0, N-2) // segment lower bound
    //     y    = y[idx] + (fidx - idx) * (y[idx+1] - y[idx])
    //     grad = (y[idx+1] - y[idx]) * inv_dx
    // For x outside [x0, xN-1], the segment is clamped (constant extrapolation
    // beyond the resample range). The raw interp() does linear extrapolation
    // off the table edges, but the resample range covers the full physical
    // domain (RH in [0, rhmax]; water in [0, max(table_water)]) so the two
    // behaviours agree wherever the simulation actually evaluates.
    constexpr int FAST_NGRID = 256;
    struct FastTable
    {
        bool built = false;
        Real64 x0 = 0.0;       // First grid point
        Real64 inv_dx = 0.0;   // Reciprocal of step size for fast index math
        Array1D<Real64> y;     // Resampled values; size FAST_NGRID (ObjexxFCL is 1-based)
    };

    struct MaterialHAMT : public Material::MaterialBase
    {
        // HAMT
        int niso = -1;                                       // Number of data points
        Array1D<Real64> isodata = Array1D<Real64>(27, 0.0);  // isotherm values
        Array1D<Real64> isorh = Array1D<Real64>(27, 0.0);    // isotherm RH values
        int nsuc = -1;                                       // Number of data points
        Array1D<Real64> sucdata = Array1D<Real64>(27, 0.0);  // suction values
        Array1D<Real64> sucwater = Array1D<Real64>(27, 0.0); // suction water values
        int nred = -1;                                       // Number of data points
        Array1D<Real64> reddata = Array1D<Real64>(27, 0.0);  // redistribution values
        Array1D<Real64> redwater = Array1D<Real64>(27, 0.0); // redistribution water values
        int nmu = -1;                                        // Number of data points
        Array1D<Real64> mudata = Array1D<Real64>(27, 0.0);   // mu values
        Array1D<Real64> murh = Array1D<Real64>(27, 0.0);     // mu rh values
        int ntc = -1;                                        // Number of data points
        Array1D<Real64> tcdata = Array1D<Real64>(27, 0.0);   // thermal conductivity values
        Array1D<Real64> tcwater = Array1D<Real64>(27, 0.0);  // thermal conductivity water values
        Real64 itemp = 10.0;                                 // initial Temperature
        Real64 irh = 0.5;                                    // Initial RH
        Real64 iwater = 0.2;                                 // Initial water content kg/kg
        int divs = 3;                                        // Number of divisions
        Real64 divsize = 0.005;                              // Average Cell Size
        int divmin = 3;                                      // Minimum number of cells
        int divmax = 10;                                     // Maximum number of cells

        // Pre-resampled property tables (built in InitHeatBalHAMT).
        // isoFast/muFast are indexed by relative humidity (x in [0, rhmax]).
        // sucFast/redFast/tcFast are indexed by water content (x in [0, max_water]).
        FastTable isoFast;
        FastTable sucFast;
        FastTable redFast;
        FastTable muFast;
        FastTable tcFast;

        MaterialHAMT() : Material::MaterialBase()
        {
            group = Material::Group::Regular;
        }
        ~MaterialHAMT() = default;
    };

    constexpr int adjmax(6); // Maximum Number of Adjacent Cells

    constexpr Real64 wdensity(1000.0); // Density of water kg.m-3
    constexpr Real64 wspech(4180.0);   // Specific Heat Capacity of Water J.kg-1.K-1 (at 20C)
    constexpr Real64 whv(2489000.0);   // Evaporation enthalpy of water J.kg-1
    constexpr Real64 qvplim(100000.0); // Maximum latent heat W
    constexpr Real64 rhmax(1.01);      // Maximum RH value

    // Solver defaults — overridden by HeatBalanceSettings:HeatAndMoistureTransfer
    constexpr int ittermax_default(150);
    constexpr Real64 convt_default(0.002);
    constexpr Real64 convphi_default(0.001);

    // Types
    struct subcell
    {
        // Members
        int matid;        // Material Id Number
        const MaterialHAMT *mat = nullptr; // Cached non-owning pointer to MaterialHAMT,
                                            // set once in InitHeatBalHAMT to avoid a
                                            // dynamic_cast in the Picard inner loop.
                                            // nullptr for boundary (non-material) cells.
        int sid;          // Surface Id Number
        Real64 Qadds;     // Additional sources of heat
        Real64 density;   // Density
        Real64 wthermalc; // Moisture Dependent Thermal Conductivity
        Real64 spech;     // Specific Heat capacity
        Real64 htc;       // Heat Transfer Coefficient
        Real64 vtc;       // Vapor Transfer Coefficient
        Real64 mu;        // Vapor Diffusion resistance Factor
        Real64 volume;    // Cell Volume
        Real64 temp;
        Real64 tempp1;
        Real64 tempp2;
        // BDF2 history: temperature at the previous-but-one timestep (T^{n-1}).
        // Maintained by UpdateHeatBalHAMT (shifted in before cell.temp ← cell.tempp1).
        // Only used when schemeType == FullyImplicitSecondOrder; ignored otherwise.
        // Sentinel: temp_prev_valid = false marks the BeginEnvrn state — the first
        // timestep after that uses backward-Euler weights (no T^{n-1} available).
        Real64 temp_prev = 0.0;
        bool temp_prev_valid = false;
        Real64 wreport;  // Water content for reporting [kg/kg dry mass]
        Real64 vpreport = 0.0; // Vapor pressure for reporting [Pa], set each timestep in UpdateHeatBalHAMT
        Real64 water;   // Water Content of cells
        Real64 vp;      // Vapor Pressure
        Real64 vpp1;    // Vapor Pressure
        Real64 vpsat;   // Saturation Vapor Pressure
        Real64 wvdc;    // Cached WVDC(tempp1, baroPress). Refreshed once per
                        // Picard iter for material cells; once per CalcHeatBalHAMT
                        // for BC cells. Avoids re-calling the pow()-based WVDC
                        // formula 2-4 times per cell-neighbor pair in the
                        // assembly inner loops.
        int nadj = 0;   // Cached count of valid neighbors (i.e. positions in
                        // adjs(1..adjmax) that hold a real cell id, not the
                        // -1 sentinel). Set once in InitHeatBalHAMT. Used by
                        // the Picard inner loops as `for ii in 1..nadj` so the
                        // sentinel check `if (adj == -1) break` can be dropped.
        Real64 rh;
        Real64 rhp1;
        Real64 rhp2;             // Relative Humidity
        // BDF2 history for relative humidity (φ^{n-1}); see notes on temp_prev above.
        Real64 rh_prev = 0.0;
        bool   rh_prev_valid = false;
        Real64 rhp;              // cell relative humidity (percent - reporting)
        Real64 dwdphi;           // Moisture storage capacity
        Real64 dw;               // Liquid transport Coefficient
        Real64 qflux;            // Heat flux at the cell's interior-facing face [W/m²], positive = exterior→interior
        Array1D<Real64> origin;  // Cell origin. The geometric centre of the cell.
        Array1D<Real64> length;  // Cell lengths
        Array1D<Real64> overlap; // Area of overlap
        Array1D<Real64> dist;    // distance between cell origins
        Array1D_int adjs;
        Array1D_int adjsl;

        // Default Constructor
        subcell()
            : matid(-1), sid(-1), Qadds(0.0), density(-1.0), wthermalc(0.0), spech(0.0), htc(-1.0), vtc(-1.0), mu(-1.0), volume(0.0), temp(0.0),
              tempp1(0.0), tempp2(0.0), wreport(0.0), water(0.0), vp(0.0), vpp1(0.0), vpsat(0.0), wvdc(0.0), nadj(0), rh(0.1), rhp1(0.1), rhp2(0.1), rhp(10.0),
              dwdphi(-1.0), dw(-1.0), qflux(0.0), origin(3, 0.0), length(3, 0.0), overlap(6, 0.0), dist(6, 0.0), adjs(6, 0), adjsl(6, 0)
        {
        }
    };

    void ManageHeatBalHAMT(EnergyPlusData &state, int const SurfNum, Real64 &SurfTempInTmp, Real64 &TempSurfOutTmp);

    void GetHeatBalHAMTInput(EnergyPlusData &state);

    void InitHeatBalHAMT(EnergyPlusData &state);

    void CalcHeatBalHAMT(EnergyPlusData &state, int const sid, Real64 &SurfTempInTmp, Real64 &TempSurfOutTmp);

    void CalcHeatBalHAMT_Thomas(EnergyPlusData &state, int const sid);

    void UpdateHeatBalHAMT(EnergyPlusData &state, int const sid);

    // Solves the tridiagonal system in-place. Modifies b and d. Arrays are 1-based.
    void thomas_solve(Array1D<Real64> &a,
                      Array1D<Real64> &b,
                      Array1D<Real64> const &c,
                      Array1D<Real64> &d,
                      Array1D<Real64> &x,
                      int N);

    void interp(int const ndata,
                const Array1D<Real64> &xx,
                const Array1D<Real64> &yy,
                Real64 const invalue,
                Real64 &outvalue,
                ObjexxFCL::Optional<Real64> outgrad = _);

    // Build the 5 pre-resampled FastTable objects on a MaterialHAMT from its raw
    // isorh/isodata/etc. tables. Called once at init.
    void BuildFastTables(MaterialHAMT &mat);

    // O(1) uniform-grid interp. Returns y; if grad != nullptr, also writes the
    // local slope. Behaviour outside [x0, x0+(NGRID-1)/inv_dx] is clamped to
    // the segment value (cf. comment on FastTable).
    inline void fast_interp(const FastTable &tbl, Real64 const x, Real64 &y, Real64 *grad = nullptr)
    {
        Real64 const fidx = (x - tbl.x0) * tbl.inv_dx;
        int idx = static_cast<int>(fidx);
        if (idx < 0) {
            idx = 0;
        } else if (idx >= FAST_NGRID - 1) {
            idx = FAST_NGRID - 2;
        }
        Real64 const frac = fidx - static_cast<Real64>(idx);
        Real64 const y0 = tbl.y(idx + 1); // ObjexxFCL Array1D is 1-based
        Real64 const y1 = tbl.y(idx + 2);
        y = y0 + frac * (y1 - y0);
        if (grad != nullptr) {
            *grad = (y1 - y0) * tbl.inv_dx;
        }
    }

    Real64 RHtoVP(EnergyPlusData &state, Real64 const RH, Real64 const Temperature);

    Real64 WVDC(Real64 const Temperature, Real64 const ambp);

    //                                 COPYRIGHT NOTICE

    //     Portions Copyright (c) University College London 2007.  All rights
    //     reserved.

    //     UCL LEGAL NOTICE
    //     Neither UCL, members of UCL nor any person or organisation acting on
    //     behalf of either:

    //     A. Makes any warranty of representation, express or implied with
    //        respect to the accuracy, completeness, or usefulness of the
    //        information contained in this program, including any warranty of
    //        merchantability or fitness of any purpose with respect to the
    //        program, or that the use of any information disclosed in this
    //        program may not infringe privately-owned rights, or

    //     B. Assumes any liability with respect to the use of, or for any and
    //        all damages resulting from the use of the program or any portion
    //        thereof or any information disclosed therein.

} // namespace HeatBalanceHAMTManager

struct HeatBalHAMTMgrData : BaseGlobalStruct
{

    Array1D_int firstcell;
    Array1D_int lastcell;
    Array1D_int Extcell;
    Array1D_int ExtRadcell;
    Array1D_int ExtConcell;
    Array1D_int ExtSkycell;
    Array1D_int ExtGrncell;
    Array1D_int Intcell;
    Array1D_int IntConcell;
    Array1D<Real64> watertot;
    Array1D<Real64> surfrh;
    Array1D<Real64> surfextrh;
    Array1D<Real64> surftemp;
    Array1D<Real64> surfexttemp;
    Array1D<Real64> surfvp;
    Array1D<Real64> extvtc;   // External Surface vapor transfer coefficient
    Array1D<Real64> intvtc;   // Internal Surface Vapor Transfer Coefficient
    Array1D_bool extvtcflag;  // External Surface vapor transfer coefficient flag
    Array1D_bool intvtcflag;  // Internal Surface Vapor Transfer Coefficient flag
    Array1D_bool MyEnvrnFlag; // Flag to reset surface properties.
    Real64 deltat = 0.0;      // time step in seconds
    int TotCellsMax = 0;      // Maximum number of cells per material
    bool latswitch = false;   // latent heat switch,
    bool rainswitch = false;  // rain switch,
    Array1D<HeatBalanceHAMTManager::subcell> cells;
    bool OneTimeFlag = true;
    int qvpErrCount = 0;
    int qvpErrReport = 0;

    // Linearization-iteration statistics.
    //   - Cumulative counters (callCount, iterTotal, iterMax) span the full
    //     simulation and are useful for diagnosing whether the linearization
    //     safety net is firing often (it should rarely fire on weakly nonlinear
    //     materials). Pre-Phase-3 these tracked "Picard" iterations; the
    //     Thomas solver now runs a 1-step linearly-implicit solve with a
    //     safety-net fallback so "linearization" is the more accurate name.
    //   - lastIterCount is per-surface: the iter count from the most recent
    //     CalcHeatBalHAMT_Thomas call. Exposed as the EP output variable
    //     "HAMT Surface Linearization Iterations", reportable per-timestep.
    long long linearizationCallCount = 0;
    long long linearizationIterTotal = 0;
    int linearizationIterMax = 0;
    Array1D<Real64> lastIterCount;  // size = TotSurfaces; 0 for non-HAMT surfaces

    // ── Phase 8: 2nd-order BC extrapolation for BDF2 ──────────────────────────
    // The implicit BDF2 solve at t^{n+1} needs BC values at t^{n+1}, but
    // EnergyPlus only exposes BC values at "now" (TempOutsideAirFD from
    // scheduled weather, spaceMAT from the zone HB).  For Thomas BE this is
    // consistent (O(Δt) BC ↔ O(Δt) solver).  For BDF2, the O(Δt) BC error
    // caps the global accuracy at O(Δt) — confirmed by the timestep_study.
    //
    // Fix: cache the previous-step BC values, then linearly extrapolate to
    // t^{n+1}:   T_BC^{n+1} ≈ 2·T_BC^n − T_BC^{n-1}   (2nd-order accurate).
    // First step of each environment falls back to the raw current value
    // (no history), matching the BE-startup convention for cell.temp_prev.
    // Phase 8a (initial): linear extrapolation 2·T^n − T^{n-1} (2nd-order, but
    //   amplifies high-frequency BC content → overshoots on rapid BC reversals).
    // Phase 8b (current):  minmod slope limiter using two history levels.
    //   slope_n   = T^n − T^{n-1}
    //   slope_nm1 = T^{n-1} − T^{n-2}
    //   limited   = minmod(slope_n, slope_nm1)  (zero at sign flips, smaller
    //                                            magnitude otherwise)
    //   T_BC^{n+1} ≈ T^n + limited
    // This drops to 1st-order at local extrema (sign flip) but stays 2nd-order
    // on smooth segments — exactly the right tradeoff for noisy/sinusoidal BC.
    Array1D<Real64> tempOutPrev;     // size = TotSurfaces; cached T_out^{n-1}
    Array1D<Real64> tempOutPrev2;    // size = TotSurfaces; cached T_out^{n-2}
    Array1D<Real64> spaceMATPrev;    // size = TotSurfaces; cached spaceMAT^{n-1}
    Array1D<Real64> spaceMATPrev2;   // size = TotSurfaces; cached spaceMAT^{n-2}
    Array1D<int>    bcHistoryDepth;  // 0=none, 1=one prev (linear), 2=two prev (limiter)

    // ── Scheme settings (populated by GetHeatBalHAMTInput) ──────────────────────
    // Default is BDF2 (2nd-order accurate, A-stable, minmod BC extrapolation).
    // BDF2 is both more accurate and at least as fast as Thomas BE at the
    // baseline mesh, and matches Thomas BE performance at finer meshes.
    // When the IDF omits HeatBalanceSettings:HeatAndMoistureTransfer entirely,
    // the input parser still applies these defaults (so users don't need to
    // include the object just to get the standard behaviour).
    HeatBalanceHAMTManager::HAMTScheme schemeType = HeatBalanceHAMTManager::HAMTScheme::FullyImplicitSecondOrder;
    // True once the input parser has run. The legacy hardwired GaussSeidel
    // meshing formula (N = int(L/divsize) + divmin) is only used when the user
    // explicitly selects GaussSeidel AND the optional N3 (Min Cells) is set —
    // see the GS branch in InitHeatBalHAMT. Everywhere else (including the
    // default no-object path), the Fourier criterion applies.
    bool settingsObjectPresent = false;
    // N1 — Space Discretization Constant. C = 1.0 corresponds to a cell ≈
    // one thermal-diffusion length per timestep (Fourier number Fo = 1).
    Real64 spaceDescritConstant = 1.0;
    // N3 / N4 — Min / Max cells per material layer. 0 means "not specified
    // by the user → no enforcement". When non-zero, the Fourier-driven cell
    // count is clamped to [HAMTdivmin, HAMTdivmax].
    int HAMTdivmin = 0;
    int HAMTdivmax = 0;
    // N2 — Boundary Layer Space Discretization Constant. Same units as N1,
    // applied to the near-face (boundary) cells only. Default 0.1 → boundary
    // cell ≈ 1/sqrt(10) of a diffusion length, finer than interior for surface
    // transients. Set to 0.0 to disable boundary refinement.
    Real64 HAMTboundaryC = 0.1;
    int HAMTittermax = HeatBalanceHAMTManager::ittermax_default;
    Real64 HAMTconvt = HeatBalanceHAMTManager::convt_default;
    Real64 HAMTconvphi = HeatBalanceHAMTManager::convphi_default;
    Real64 HAMTrelaxFactor = 1.0;

    // ── Iteration-cap recurring warning state ─────────────────────────────────
    // Two-stage pattern (matches qvpErrCount / qvpErrReport):
    //   - First 16 occurrences → ShowWarningError with surface name + timestamp.
    //   - After that          → ShowRecurringWarningErrorAtEnd (aggregated count).
    // GS and Thomas paths use separate counters so the report index is stable.
    int gsIterCapErrCount = 0;      // first-stage occurrence counter (GS path)
    int gsIterCapErrReport = 0;     // ShowRecurringWarningErrorAtEnd index (GS path)
    int thomasIterCapErrCount = 0;  // first-stage occurrence counter (Thomas / BDF2 path)
    int thomasIterCapErrReport = 0; // ShowRecurringWarningErrorAtEnd index (Thomas / BDF2 path)

    // Outside face vapor pressure [Pa] — symmetric partner to surfvp.
    // Set each timestep in UpdateHeatBalHAMT; exposed as output variable.
    Array1D<Real64> surfoutvp;

    // Linearization safety thresholds (Phase 3b, IDD N9/N10).
    // After the first linearized solve, if |ΔT| > linearizationSafetyTemp OR
    // |Δφ| > linearizationSafetyRH (per-timestep cell-level state change from
    // entering values), the safety net fires iter 2 with refreshed properties.
    // Conservative defaults: 2°C and 5 pp RH — well above the convergence
    // tolerances above so iter 2 only fires for genuinely large transients.
    // Setting these to very small values approaches Phase 0 (iter 2 always fires);
    // setting them very large approaches pure Phase 3 (iter 2 never fires).
    Real64 linearizationSafetyTemp = 2.0;
    Real64 linearizationSafetyRH = 0.05;

    // ── Thomas working arrays (allocated to TotCellsMax in InitHeatBalHAMT) ────
    Array1D<Real64> thomas_a;
    Array1D<Real64> thomas_b;
    Array1D<Real64> thomas_c;
    Array1D<Real64> thomas_d;
    Array1D<Real64> thomas_x;
    // Previous-iterate snapshots for Picard convergence check. Allocated once
    // to TotCellsMax in InitHeatBalHAMT (was per-call Array1D<Real64>(N) before).
    Array1D<Real64> thomas_tempp1_prev;
    Array1D<Real64> thomas_rhp1_prev;

    void init_constant_state([[maybe_unused]] EnergyPlusData &state) override
    {
    }

    void init_state([[maybe_unused]] EnergyPlusData &state) override
    {
    }

    void clear_state() override
    {
        this->OneTimeFlag = true;
        this->qvpErrCount = 0;
        this->qvpErrReport = 0;
        this->settingsObjectPresent = false;
        this->thomas_a.deallocate();
        this->thomas_b.deallocate();
        this->thomas_c.deallocate();
        this->thomas_d.deallocate();
        this->thomas_x.deallocate();
        this->thomas_tempp1_prev.deallocate();
        this->thomas_rhp1_prev.deallocate();
        this->lastIterCount.deallocate();
        this->surfoutvp.deallocate();
        this->gsIterCapErrCount = 0;
        this->gsIterCapErrReport = 0;
        this->thomasIterCapErrCount = 0;
        this->thomasIterCapErrReport = 0;
        this->linearizationCallCount = 0;
        this->linearizationIterTotal = 0;
        this->linearizationIterMax = 0;
    }
};

} // namespace EnergyPlus

#endif
