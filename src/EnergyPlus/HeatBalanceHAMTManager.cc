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

// C++ Headers
#include <cmath>
#include <format>
#include <fstream>
#include <string>

// ObjexxFCL Headers
#include <ObjexxFCL/Fmath.hh>
#include <ObjexxFCL/member.functions.hh>

// EnergyPlus Headers
#include <EnergyPlus/Construction.hh>
#include <EnergyPlus/Data/EnergyPlusData.hh>
#include <EnergyPlus/DataEnvironment.hh>
#include <EnergyPlus/DataHeatBalSurface.hh>
#include <EnergyPlus/DataHeatBalance.hh>
#include <EnergyPlus/DataIPShortCuts.hh>
#include <EnergyPlus/DataMoistureBalance.hh>
#include <EnergyPlus/DataSurfaces.hh>
#include <EnergyPlus/DisplayRoutines.hh>
#include <EnergyPlus/General.hh>
#include <EnergyPlus/HeatBalanceHAMTManager.hh>
#include <EnergyPlus/InputProcessing/InputProcessor.hh>
#include <EnergyPlus/Material.hh>
#include <EnergyPlus/OutputProcessor.hh>
#include <EnergyPlus/Psychrometrics.hh>
#include <EnergyPlus/UtilityRoutines.hh>
#include <EnergyPlus/ZoneTempPredictorCorrector.hh>

namespace EnergyPlus {

namespace HeatBalanceHAMTManager {

    // MODULE INFORMATION:
    //       AUTHOR         Phillip Biddulph
    //       DATE WRITTEN   June 2008
    //       MODIFIED
    //       Aug 2009: Phillip Biddulph: Bug fixes to make sure HAMT can cope with data limits
    //       May 2026: Gabriel Flechas, PhD: Added two direct (Thomas/TDMA) time-integration
    //                 schemes alongside the original Gauss-Seidel solver, selectable through the
    //                 new HeatBalanceSettings:HeatAndMoistureTransfer object:
    //                   - FullyImplicitFirstOrder-Thomas : backward-Euler, O(dt)
    //                   - FullyImplicitSecondOrder-Thomas: BDF2 (second-order Backward
    //                     Differentiation Formula), O(dt^2), now the default
    //                 plus physics-based (Fourier-number) cell meshing, an outer Picard loop with
    //                 linearization safety net, minmod-limited boundary-condition extrapolation for
    //                 BDF2, per-cell vapor-pressure/heat-flux output variables, and iteration-cap
    //                 recurring warnings. The original Gauss-Seidel scheme (below, clearly marked
    //                 "LEGACY GAUSS-SEIDEL PATH") is preserved unchanged and remains selectable.
    //                 Developed with the assistance of Anthropic Claude models
    //                 (Sonnet 4.6, Opus 4.7, Opus 4.8).
    //       RE-ENGINEERED

    // PURPOSE OF THIS MODULE:
    // Calculate, record and report the one dimensional heat and moisture transfer
    // through a surface given the material composition of the building surface and
    // the external and internal Temperatures and Relative Humidities.

    // METHODOLOGY EMPLOYED:
    // Each surface is split into "cells", where all characteristics are initialised.
    // Cells are matched and links created in the initialisation routine.
    // The internal and external "surfaces" of the surface are virtual cells to allow for the
    // input of heat and vapor via heat transfer coefficients, radiation,
    // and vapor transfer coefficients
    // Uses Forward (implicit) finite difference algorithm. Heat transfer is calculated first,
    // with the option of including the latent heat, then liquid and vapor transfer. The process is ittereated.
    // Once the temperatures have converged the internal surface
    // temperature and vapor densities are passed back to EnergyPlus.

    // Temperatures and relative humidities are updated once EnergyPlus has checked that
    // the zone temperatures have converged.

    // REFERENCES:
    // Governing heat/moisture transport equations and material physics (all schemes):
    //   Kunzel, H.M. (1995) Simultaneous Heat and Moisture Transport in Building Components.
    //     One- and two-dimensional calculation using simple parameters. IRB Verlag 1995
    //   Holman, J.P. (2002) Heat Transfer, Ninth Edition. McGraw-Hill
    //   Winterton, R.H.S. (1997) Heat Transfer. (Oxford Chemistry Primers; 50) Oxford University Press
    //   Kumar Kumaran, M. (1996) IEA ANNEX 24, Final Report, Volume 3
    // Time-integration and solver methods added May 2026 (Thomas/BDF2 schemes):
    //   Thomas, L.H. (1949) Elliptic Problems in Linear Difference Equations over a Network.
    //     Watson Sci. Comput. Lab. Report, Columbia University.  [TDMA direct tridiagonal solve]
    //   Curtiss, C.F. & Hirschfelder, J.O. (1952) Integration of Stiff Equations.
    //     PNAS 38(3):235-243.  [origin of the BDF family]
    //   Ascher, U.M. & Petzold, L.R. (1998) Computer Methods for ODEs and DAEs. SIAM.
    //     Sec. 5.1-5.2, BDF order-2 weights (1.5, -2, 0.5) and A/L-stability.
    //   Harten, A. (1983) High Resolution Schemes for Hyperbolic Conservation Laws.
    //     J. Comput. Phys. 49:357-393.  [minmod / total-variation-diminishing limiter]
    //   LeVeque, R.J. (2002) Finite Volume Methods for Hyperbolic Problems. Cambridge.
    //     Sec. 6.9, minmod slope limiter used here for BDF2 boundary-condition extrapolation.

    // USE STATEMENTS:

    // Using/Aliasing
    using namespace DataSurfaces;
    using DataHeatBalSurface::MinSurfaceTempLimit;
    using DataHeatBalSurface::MinSurfaceTempLimitBeforeFatal;
    using namespace DataHeatBalance;
    using namespace Psychrometrics;

    void ManageHeatBalHAMT(EnergyPlusData &state, int const SurfNum, Real64 &SurfTempInTmp, Real64 &TempSurfOutTmp)
    {

        // SUBROUTINE INFORMATION:
        //       AUTHOR         Phillip Biddulph
        //       DATE WRITTEN   June 2008
        //       MODIFIED       na
        //       RE-ENGINEERED  na

        // PURPOSE OF THIS SUBROUTINE:
        // Manages the Heat and Moisture Transfer calculations.

        // SUBROUTINE LOCAL VARIABLE DECLARATIONS:
        if (state.dataHeatBalHAMTMgr->OneTimeFlag) {
            state.dataHeatBalHAMTMgr->OneTimeFlag = false;
            DisplayString(state, "Initialising Heat and Moisture Transfer Model");
            GetHeatBalHAMTInput(state);
            InitHeatBalHAMT(state);
        }

        CalcHeatBalHAMT(state, SurfNum, SurfTempInTmp, TempSurfOutTmp);
    }

    void GetHeatBalHAMTInput(EnergyPlusData &state)
    {

        // SUBROUTINE INFORMATION:
        //       AUTHOR         Phillip Biddulph
        //       DATE WRITTEN   June 2008
        //       MODIFIED       na
        //       RE-ENGINEERED  na

        // PURPOSE OF THIS SUBROUTINE:
        // gets input for the HAMT model
        static constexpr std::string_view routineName = "GetHeatBalHAMTInput";

        // SUBROUTINE PARAMETER DEFINITIONS:
        static std::string const cHAMTObject1("MaterialProperty:HeatAndMoistureTransfer:Settings");
        static std::string const cHAMTObject2("MaterialProperty:HeatAndMoistureTransfer:SorptionIsotherm");
        static std::string const cHAMTObject3("MaterialProperty:HeatAndMoistureTransfer:Suction");
        static std::string const cHAMTObject4("MaterialProperty:HeatAndMoistureTransfer:Redistribution");
        static std::string const cHAMTObject5("MaterialProperty:HeatAndMoistureTransfer:Diffusion");
        static std::string const cHAMTObject6("MaterialProperty:HeatAndMoistureTransfer:ThermalConductivity");
        static std::string const cHAMTObject7("SurfaceProperties:VaporCoefficients");
        static std::string const cHAMTSettings("HeatBalanceSettings:HeatAndMoistureTransfer");

        // SUBROUTINE LOCAL VARIABLE DECLARATIONS:

        Array1D_string AlphaArray;
        Array1D_string cAlphaFieldNames;
        Array1D_string cNumericFieldNames;

        Array1D_bool lAlphaBlanks;
        Array1D_bool lNumericBlanks;

        Array1D<Real64> NumArray;

        Real64 avdata;

        int MaxNums;
        int MaxAlphas;
        int NumParams;
        int NumNums;
        int NumAlphas;
        int status;
        int Numid;

        int HAMTitems;

        bool ErrorsFound;

        auto &s_ip = state.dataInputProcessing->inputProcessor;
        auto &s_mat = state.dataMaterial;

        state.dataHeatBalHAMTMgr->watertot.allocate(state.dataSurface->TotSurfaces);
        state.dataHeatBalHAMTMgr->surfrh.allocate(state.dataSurface->TotSurfaces);
        state.dataHeatBalHAMTMgr->surfextrh.allocate(state.dataSurface->TotSurfaces);
        state.dataHeatBalHAMTMgr->surftemp.allocate(state.dataSurface->TotSurfaces);
        state.dataHeatBalHAMTMgr->surfexttemp.allocate(state.dataSurface->TotSurfaces);
        state.dataHeatBalHAMTMgr->surfvp.allocate(state.dataSurface->TotSurfaces);
        state.dataHeatBalHAMTMgr->surfoutvp.allocate(state.dataSurface->TotSurfaces);
        state.dataHeatBalHAMTMgr->lastIterCount.allocate(state.dataSurface->TotSurfaces);
        state.dataHeatBalHAMTMgr->lastIterCount = 0.0;

        // Phase 8: BC extrapolation history (see comment in .hh).
        state.dataHeatBalHAMTMgr->tempOutPrev.allocate(state.dataSurface->TotSurfaces);
        state.dataHeatBalHAMTMgr->tempOutPrev2.allocate(state.dataSurface->TotSurfaces);
        state.dataHeatBalHAMTMgr->spaceMATPrev.allocate(state.dataSurface->TotSurfaces);
        state.dataHeatBalHAMTMgr->spaceMATPrev2.allocate(state.dataSurface->TotSurfaces);
        state.dataHeatBalHAMTMgr->bcHistoryDepth.allocate(state.dataSurface->TotSurfaces);
        state.dataHeatBalHAMTMgr->tempOutPrev    = 0.0;
        state.dataHeatBalHAMTMgr->tempOutPrev2   = 0.0;
        state.dataHeatBalHAMTMgr->spaceMATPrev   = 0.0;
        state.dataHeatBalHAMTMgr->spaceMATPrev2  = 0.0;
        state.dataHeatBalHAMTMgr->bcHistoryDepth = 0;

        state.dataHeatBalHAMTMgr->firstcell.allocate(state.dataSurface->TotSurfaces);
        state.dataHeatBalHAMTMgr->lastcell.allocate(state.dataSurface->TotSurfaces);
        state.dataHeatBalHAMTMgr->Extcell.allocate(state.dataSurface->TotSurfaces);
        state.dataHeatBalHAMTMgr->ExtRadcell.allocate(state.dataSurface->TotSurfaces);
        state.dataHeatBalHAMTMgr->ExtConcell.allocate(state.dataSurface->TotSurfaces);
        state.dataHeatBalHAMTMgr->ExtSkycell.allocate(state.dataSurface->TotSurfaces);
        state.dataHeatBalHAMTMgr->ExtGrncell.allocate(state.dataSurface->TotSurfaces);
        state.dataHeatBalHAMTMgr->Intcell.allocate(state.dataSurface->TotSurfaces);
        state.dataHeatBalHAMTMgr->IntConcell.allocate(state.dataSurface->TotSurfaces);

        state.dataHeatBalHAMTMgr->extvtc.allocate(state.dataSurface->TotSurfaces);
        state.dataHeatBalHAMTMgr->intvtc.allocate(state.dataSurface->TotSurfaces);
        state.dataHeatBalHAMTMgr->extvtcflag.allocate(state.dataSurface->TotSurfaces);
        state.dataHeatBalHAMTMgr->intvtcflag.allocate(state.dataSurface->TotSurfaces);
        state.dataHeatBalHAMTMgr->MyEnvrnFlag.allocate(state.dataSurface->TotSurfaces);

        state.dataHeatBalHAMTMgr->extvtc = -1.0;
        state.dataHeatBalHAMTMgr->intvtc = -1.0;
        state.dataHeatBalHAMTMgr->extvtcflag = false;
        state.dataHeatBalHAMTMgr->intvtcflag = false;
        state.dataHeatBalHAMTMgr->MyEnvrnFlag = true;

        state.dataHeatBalHAMTMgr->latswitch = true;
        state.dataHeatBalHAMTMgr->rainswitch = true;

        MaxAlphas = 0;
        MaxNums = 0;
        s_ip->getObjectDefMaxArgs(state, cHAMTObject1, NumParams, NumAlphas, NumNums);
        MaxAlphas = max(MaxAlphas, NumAlphas);
        MaxNums = max(MaxNums, NumNums);
        s_ip->getObjectDefMaxArgs(state, cHAMTObject2, NumParams, NumAlphas, NumNums);
        MaxAlphas = max(MaxAlphas, NumAlphas);
        MaxNums = max(MaxNums, NumNums);
        s_ip->getObjectDefMaxArgs(state, cHAMTObject3, NumParams, NumAlphas, NumNums);
        MaxAlphas = max(MaxAlphas, NumAlphas);
        MaxNums = max(MaxNums, NumNums);
        s_ip->getObjectDefMaxArgs(state, cHAMTObject4, NumParams, NumAlphas, NumNums);
        MaxAlphas = max(MaxAlphas, NumAlphas);
        MaxNums = max(MaxNums, NumNums);
        s_ip->getObjectDefMaxArgs(state, cHAMTObject5, NumParams, NumAlphas, NumNums);
        MaxAlphas = max(MaxAlphas, NumAlphas);
        MaxNums = max(MaxNums, NumNums);
        s_ip->getObjectDefMaxArgs(state, cHAMTObject6, NumParams, NumAlphas, NumNums);
        MaxAlphas = max(MaxAlphas, NumAlphas);
        MaxNums = max(MaxNums, NumNums);
        s_ip->getObjectDefMaxArgs(state, cHAMTObject7, NumParams, NumAlphas, NumNums);
        MaxAlphas = max(MaxAlphas, NumAlphas);
        MaxNums = max(MaxNums, NumNums);
        s_ip->getObjectDefMaxArgs(state, cHAMTSettings, NumParams, NumAlphas, NumNums);
        MaxAlphas = max(MaxAlphas, NumAlphas);
        MaxNums = max(MaxNums, NumNums);

        ErrorsFound = false;

        AlphaArray.allocate(MaxAlphas);
        cAlphaFieldNames.allocate(MaxAlphas);
        cNumericFieldNames.allocate(MaxNums);
        NumArray.dimension(MaxNums, 0.0);
        lAlphaBlanks.dimension(MaxAlphas, false);
        lNumericBlanks.dimension(MaxNums, false);

        // ── HeatBalanceSettings:HeatAndMoistureTransfer ──────────────────────────
        // If the object is absent, the defaults declared in HeatBalHAMTMgrData
        // already match what a fully-blank object would produce:
        //   schemeType         = FullyImplicitSecondOrder (BDF2, most accurate + fastest)
        //   spaceDescritConstant = 1.0  (interior)
        //   HAMTboundaryC      = 0.1    (boundary, finer)
        //   HAMTdivmin/divmax  = 0      (not enforced)
        //   HAMTittermax       = 150
        //   HAMTconvt/convphi  = 0.002 / 0.001
        //   HAMTrelaxFactor    = 1.0
        //   linearizationSafetyTemp / RH = 2.0 / 0.05
        // We still mark settingsObjectPresent = true so the Fourier-based mesh
        // formula runs unconditionally — the legacy hardwired GS divsize formula
        // is no longer the default.
        auto &s_hbh = state.dataHeatBalHAMTMgr;
        s_hbh->settingsObjectPresent = true;

        if (s_ip->getNumObjectsFound(state, cHAMTSettings) > 0) {
            s_ip->getObjectItem(state,
                                cHAMTSettings,
                                1,
                                AlphaArray,
                                NumAlphas,
                                NumArray,
                                NumNums,
                                status,
                                lNumericBlanks,
                                lAlphaBlanks,
                                cAlphaFieldNames,
                                cNumericFieldNames);

            // A1 — Difference Scheme
            if (!lAlphaBlanks(1)) {
                std::string const scheme = Util::makeUPPER(AlphaArray(1));
                if (scheme == "FULLYIMPLICITFIRSTORDER-THOMAS") {
                    s_hbh->schemeType = HAMTScheme::FullyImplicitThomas;
                } else if (scheme == "FULLYIMPLICITSECONDORDER-THOMAS") {
                    s_hbh->schemeType = HAMTScheme::FullyImplicitSecondOrder;
                } else if (scheme == "FULLYIMPLICITFIRSTORDER-GAUSSSEIDEL") {
                    s_hbh->schemeType = HAMTScheme::GaussSeidel;
                } else {
                    ShowSevereError(
                        state,
                        EnergyPlus::format("{}: invalid Difference Scheme \"{}\".", cHAMTSettings, AlphaArray(1)));
                    ErrorsFound = true;
                }
            }
            // (else: scheme remains at its struct default, FullyImplicitThomas)

            // Numeric parameters, in the new field order:
            //   N1 = Space Discretization Constant         (interior)
            //   N2 = Boundary Layer Space Discretization Constant (was N8)
            //   N3 = Minimum Cells Per Material Layer      (optional, was N2)
            //   N4 = Maximum Cells Per Material Layer      (optional, was N3)
            //   N5 = Maximum Iterations
            //   N6 = Temperature Convergence Threshold
            //   N7 = Relative Humidity Convergence Threshold
            //   N8 = Relaxation Factor
            //   N9 = Linearization Safety Temperature Threshold
            //   N10 = Linearization Safety Relative Humidity Threshold
            if (!lNumericBlanks(1))  s_hbh->spaceDescritConstant     = NumArray(1);
            if (!lNumericBlanks(2))  s_hbh->HAMTboundaryC            = NumArray(2);
            if (!lNumericBlanks(3))  s_hbh->HAMTdivmin               = static_cast<int>(NumArray(3));
            if (!lNumericBlanks(4))  s_hbh->HAMTdivmax               = static_cast<int>(NumArray(4));
            if (!lNumericBlanks(5))  s_hbh->HAMTittermax             = static_cast<int>(NumArray(5));
            if (!lNumericBlanks(6))  s_hbh->HAMTconvt                = NumArray(6);
            if (!lNumericBlanks(7))  s_hbh->HAMTconvphi              = NumArray(7);
            if (!lNumericBlanks(8))  s_hbh->HAMTrelaxFactor          = NumArray(8);
            if (!lNumericBlanks(9))  s_hbh->linearizationSafetyTemp  = NumArray(9);
            if (!lNumericBlanks(10)) s_hbh->linearizationSafetyRH    = NumArray(10);
        }
        // Note: if the user explicitly selects FullyImplicitFirstOrder-GaussSeidel
        // here, the meshing still uses the new Fourier criterion (with N1/N2
        // constants). The original divsize formula is no longer reachable; users
        // who need it would have to specify it as a small-N3/large-N4 combination.

        HAMTitems = s_ip->getNumObjectsFound(state, cHAMTObject1); // MaterialProperty:HeatAndMoistureTransfer:Settings
        for (int item = 1; item <= HAMTitems; ++item) {
            s_ip->getObjectItem(state,
                                cHAMTObject1,
                                item,
                                AlphaArray,
                                NumAlphas,
                                NumArray,
                                NumNums,
                                status,
                                lNumericBlanks,
                                lAlphaBlanks,
                                cAlphaFieldNames,
                                cNumericFieldNames);

            ErrorObjectHeader eoh{routineName, cHAMTObject1, AlphaArray(1)};
            int matNum = Material::GetMaterialNum(state, AlphaArray(1));

            if (matNum == 0) {
                ShowSevereItemNotFound(state, eoh, cAlphaFieldNames(1), AlphaArray(1));
                ShowContinueError(state, "The basic material must be defined in addition to specifying HeatAndMoistureTransfer properties.");
                ErrorsFound = true;
                continue;
            }

            auto *mat = s_mat->materials(matNum);

            if (mat->group != Material::Group::Regular) {
                ShowSevereCustom(state, eoh, EnergyPlus::format("{} = \"{}\" is not a regular material.", cAlphaFieldNames(1), AlphaArray(1)));
                ErrorsFound = true;
                continue;
            }

            if (mat->ROnly) {
                ShowWarningError(
                    state,
                    EnergyPlus::format("{} {}=\"{}\" is defined as an R-only value material.", cHAMTObject1, cAlphaFieldNames(1), AlphaArray(1)));
                continue;
            }

            auto *matHAMT = new MaterialHAMT;
            matHAMT->Material::MaterialBase::operator=(*mat); // deep copy

            delete mat;
            s_mat->materials(matNum) = matHAMT;

            matHAMT->hasHAMT = true;
            matHAMT->Porosity = NumArray(1);
            matHAMT->iwater = NumArray(2);
        }

        HAMTitems = s_ip->getNumObjectsFound(state, cHAMTObject2); // MaterialProperty:HeatAndMoistureTransfer:SorptionIsotherm
        for (int item = 1; item <= HAMTitems; ++item) {
            s_ip->getObjectItem(state,
                                cHAMTObject2,
                                item,
                                AlphaArray,
                                NumAlphas,
                                NumArray,
                                NumNums,
                                status,
                                lNumericBlanks,
                                lAlphaBlanks,
                                cAlphaFieldNames,
                                cNumericFieldNames);

            ErrorObjectHeader eoh{routineName, cHAMTObject2, AlphaArray(1)};
            int matNum = Material::GetMaterialNum(state, AlphaArray(1));

            if (matNum == 0) {
                ShowSevereItemNotFound(state, eoh, cAlphaFieldNames(1), AlphaArray(1));
                ShowContinueError(state, "The basic material must be defined in addition to specifying HeatAndMoistureTransfer properties.");
                ErrorsFound = true;
                continue;
            }

            auto *mat = s_mat->materials(matNum);
            if (!mat->hasHAMT) {
                ShowSevereCustom(
                    state, eoh, EnergyPlus::format("{} is not defined for {} = \"{}\"", cHAMTObject1, cAlphaFieldNames(1), AlphaArray(1)));
                ErrorsFound = true;
                continue;
            }

            auto *matHAMT = dynamic_cast<MaterialHAMT *>(mat);
            assert(matHAMT != nullptr);

            Numid = 1;

            matHAMT->niso = int(NumArray(Numid));

            for (int iso = 1; iso <= matHAMT->niso; ++iso) {
                matHAMT->isorh(iso) = NumArray(++Numid);
                matHAMT->isodata(iso) = NumArray(++Numid);
            }

            ++matHAMT->niso;
            matHAMT->isorh(matHAMT->niso) = rhmax;
            matHAMT->isodata(matHAMT->niso) = matHAMT->Porosity * wdensity;

            ++matHAMT->niso;
            matHAMT->isorh(matHAMT->niso) = 0.0;
            matHAMT->isodata(matHAMT->niso) = 0.0;

            // check the isotherm

            // - First sort
            for (int jj = 1; jj <= matHAMT->niso - 1; ++jj) {
                for (int ii = jj + 1; ii <= matHAMT->niso; ++ii) {
                    if (matHAMT->isorh(jj) > matHAMT->isorh(ii)) {

                        Real64 dumrh = matHAMT->isorh(jj);
                        Real64 dumdata = matHAMT->isodata(jj);

                        matHAMT->isorh(jj) = matHAMT->isorh(ii);
                        matHAMT->isodata(jj) = matHAMT->isodata(ii);

                        matHAMT->isorh(ii) = dumrh;
                        matHAMT->isodata(ii) = dumdata;
                    }
                }
            }

            //- Now make sure the data rises
            bool isoerrrise = false;
            for (int ii = 1; ii <= 100; ++ii) {
                bool avflag = true;
                for (int jj = 1; jj <= matHAMT->niso - 1; ++jj) {
                    if (matHAMT->isodata(jj) > matHAMT->isodata(jj + 1)) {
                        isoerrrise = true;
                        avdata = (matHAMT->isodata(jj) + matHAMT->isodata(jj + 1)) / 2.0;
                        matHAMT->isodata(jj) = avdata;
                        matHAMT->isodata(jj + 1) = avdata;
                        avflag = false;
                    }
                }
                if (avflag) {
                    break;
                }
            }
            if (isoerrrise) {
                ShowWarningError(state, EnergyPlus::format("{}: data not rising - Check material {}", cHAMTObject2, matHAMT->Name));
                ShowContinueError(state, "Isotherm data has been fixed, and the simulation continues.");
            }
        }

        HAMTitems = s_ip->getNumObjectsFound(state, cHAMTObject3); // MaterialProperty:HeatAndMoistureTransfer:Suction
        for (int item = 1; item <= HAMTitems; ++item) {
            s_ip->getObjectItem(state,
                                cHAMTObject3,
                                item,
                                AlphaArray,
                                NumAlphas,
                                NumArray,
                                NumNums,
                                status,
                                lNumericBlanks,
                                lAlphaBlanks,
                                cAlphaFieldNames,
                                cNumericFieldNames);

            ErrorObjectHeader eoh{routineName, cHAMTObject3, AlphaArray(1)};
            int matNum = Material::GetMaterialNum(state, AlphaArray(1));

            if (matNum == 0) {
                ShowSevereItemNotFound(state, eoh, cAlphaFieldNames(1), AlphaArray(1));
                ShowContinueError(state, "The basic material must be defined in addition to specifying HeatAndMoistureTransfer properties.");
                ErrorsFound = true;
                continue;
            }

            auto *mat = s_mat->materials(matNum);
            if (!mat->hasHAMT) {
                ShowSevereCustom(
                    state, eoh, EnergyPlus::format("{} is not defined for {} = \"{}\"", cHAMTObject1, cAlphaFieldNames(1), AlphaArray(1)));
                ErrorsFound = true;
                continue;
            }

            auto *matHAMT = dynamic_cast<MaterialHAMT *>(mat);
            assert(matHAMT != nullptr);

            Numid = 1;

            matHAMT->nsuc = NumArray(Numid);
            for (int suc = 1; suc <= matHAMT->nsuc; ++suc) {
                matHAMT->sucwater(suc) = NumArray(++Numid);
                matHAMT->sucdata(suc) = NumArray(++Numid);
            }

            ++matHAMT->nsuc;
            matHAMT->sucwater(matHAMT->nsuc) = matHAMT->isodata(matHAMT->niso);
            matHAMT->sucdata(matHAMT->nsuc) = matHAMT->sucdata(matHAMT->nsuc - 1);
        }

        HAMTitems = s_ip->getNumObjectsFound(state, cHAMTObject4); // MaterialProperty:HeatAndMoistureTransfer:Redistribution
        for (int item = 1; item <= HAMTitems; ++item) {
            s_ip->getObjectItem(state,
                                cHAMTObject4,
                                item,
                                AlphaArray,
                                NumAlphas,
                                NumArray,
                                NumNums,
                                status,
                                lNumericBlanks,
                                lAlphaBlanks,
                                cAlphaFieldNames,
                                cNumericFieldNames);

            ErrorObjectHeader eoh{routineName, cHAMTObject4, AlphaArray(1)};
            int matNum = Material::GetMaterialNum(state, AlphaArray(1));
            if (matNum == 0) {
                ShowSevereItemNotFound(state, eoh, cAlphaFieldNames(1), AlphaArray(1));
                ShowContinueError(state, "The basic material must be defined in addition to specifying HeatAndMoistureTransfer properties.");
                ErrorsFound = true;
                continue;
            }

            auto *mat = s_mat->materials(matNum);
            if (!mat->hasHAMT) {
                ShowSevereCustom(
                    state, eoh, EnergyPlus::format("{} is not defined for {} = \"{}\"", cHAMTObject1, cAlphaFieldNames(1), AlphaArray(1)));
                ErrorsFound = true;
                continue;
            }

            auto *matHAMT = dynamic_cast<MaterialHAMT *>(mat);
            assert(matHAMT != nullptr);

            Numid = 1;

            matHAMT->nred = NumArray(Numid);
            for (int red = 1; red <= matHAMT->nred; ++red) {
                matHAMT->redwater(red) = NumArray(++Numid);
                matHAMT->reddata(red) = NumArray(++Numid);
            }

            ++matHAMT->nred;
            matHAMT->redwater(matHAMT->nred) = matHAMT->isodata(matHAMT->niso);
            matHAMT->reddata(matHAMT->nred) = matHAMT->reddata(matHAMT->nred - 1);
        }

        HAMTitems = s_ip->getNumObjectsFound(state, cHAMTObject5); // MaterialProperty:HeatAndMoistureTransfer:Diffusion
        for (int item = 1; item <= HAMTitems; ++item) {
            s_ip->getObjectItem(state,
                                cHAMTObject5,
                                item,
                                AlphaArray,
                                NumAlphas,
                                NumArray,
                                NumNums,
                                status,
                                lNumericBlanks,
                                lAlphaBlanks,
                                cAlphaFieldNames,
                                cNumericFieldNames);

            ErrorObjectHeader eoh{routineName, cHAMTObject5, AlphaArray(1)};
            int matNum = Material::GetMaterialNum(state, AlphaArray(1));
            if (matNum == 0) {
                ShowSevereItemNotFound(state, eoh, cAlphaFieldNames(1), AlphaArray(1));
                ShowContinueError(state, "The basic material must be defined in addition to specifying HeatAndMoistureTransfer properties.");
                ErrorsFound = true;
                continue;
            }

            auto *mat = s_mat->materials(matNum);
            if (!mat->hasHAMT) {
                ShowSevereCustom(
                    state, eoh, EnergyPlus::format("{} is not defined for {} = \"{}\"", cHAMTObject1, cAlphaFieldNames(1), AlphaArray(1)));
                ErrorsFound = true;
                continue;
            }

            auto *matHAMT = dynamic_cast<MaterialHAMT *>(mat);
            assert(matHAMT != nullptr);

            Numid = 1;

            matHAMT->nmu = NumArray(Numid);
            if (matHAMT->nmu > 0) {
                for (int mu = 1; mu <= matHAMT->nmu; ++mu) {
                    matHAMT->murh(mu) = NumArray(++Numid);
                    matHAMT->mudata(mu) = NumArray(++Numid);
                }

                ++matHAMT->nmu;
                matHAMT->murh(matHAMT->nmu) = matHAMT->isorh(matHAMT->niso);
                matHAMT->mudata(matHAMT->nmu) = matHAMT->mudata(matHAMT->nmu - 1);
            }
        }

        HAMTitems = s_ip->getNumObjectsFound(state, cHAMTObject6); // MaterialProperty:HeatAndMoistureTransfer:ThermalConductivity
        for (int item = 1; item <= HAMTitems; ++item) {
            s_ip->getObjectItem(state,
                                cHAMTObject6,
                                item,
                                AlphaArray,
                                NumAlphas,
                                NumArray,
                                NumNums,
                                status,
                                lNumericBlanks,
                                lAlphaBlanks,
                                cAlphaFieldNames,
                                cNumericFieldNames);

            ErrorObjectHeader eoh{routineName, cHAMTObject6, AlphaArray(1)};
            int matNum = Material::GetMaterialNum(state, AlphaArray(1));
            if (matNum == 0) {
                ShowSevereItemNotFound(state, eoh, cAlphaFieldNames(1), AlphaArray(1));
                ShowContinueError(state, "The basic material must be defined in addition to specifying HeatAndMoistureTransfer properties.");
                ErrorsFound = true;
                continue;
            }

            auto *mat = s_mat->materials(matNum);
            if (!mat->hasHAMT) {
                ShowSevereCustom(
                    state, eoh, EnergyPlus::format("{} is not defined for {} = \"{}\"", cHAMTObject1, cAlphaFieldNames(1), AlphaArray(1)));
                ErrorsFound = true;
                continue;
            }

            auto *matHAMT = dynamic_cast<MaterialHAMT *>(mat);
            assert(matHAMT != nullptr);

            Numid = 1;

            matHAMT->ntc = NumArray(Numid);
            if (matHAMT->ntc > 0) {
                for (int tc = 1; tc <= matHAMT->ntc; ++tc) {
                    ++Numid;
                    matHAMT->tcwater(tc) = NumArray(Numid);
                    ++Numid;
                    matHAMT->tcdata(tc) = NumArray(Numid);
                }

                ++matHAMT->ntc;
                matHAMT->tcwater(matHAMT->ntc) = matHAMT->isodata(matHAMT->niso);
                matHAMT->tcdata(matHAMT->ntc) = matHAMT->tcdata(matHAMT->ntc - 1);
            }
        }

        // Vapor Transfer coefficients
        HAMTitems = s_ip->getNumObjectsFound(state, cHAMTObject7); // SurfaceProperties:VaporCoefficients
        for (int item = 1; item <= HAMTitems; ++item) {
            s_ip->getObjectItem(state,
                                cHAMTObject7,
                                item,
                                AlphaArray,
                                NumAlphas,
                                NumArray,
                                NumNums,
                                status,
                                lNumericBlanks,
                                lAlphaBlanks,
                                cAlphaFieldNames,
                                cNumericFieldNames);

            ErrorObjectHeader eoh{routineName, cHAMTObject7, AlphaArray(1)};
            int vtcsid = Util::FindItemInList(AlphaArray(1), state.dataSurface->Surface);
            if (vtcsid == 0) {
                ShowSevereItemNotFound(state, eoh, cAlphaFieldNames(1), AlphaArray(1));
                ShowContinueError(state, "The basic material must be defined in addition to specifying HeatAndMoistureTransfer properties.");
                ErrorsFound = true;
                continue;
            }

            if (AlphaArray(2) == "YES") {
                state.dataHeatBalHAMTMgr->extvtcflag(vtcsid) = true;
                state.dataHeatBalHAMTMgr->extvtc(vtcsid) = NumArray(1);
            }

            if (AlphaArray(3) == "YES") {
                state.dataHeatBalHAMTMgr->intvtcflag(vtcsid) = true;
                state.dataHeatBalHAMTMgr->intvtc(vtcsid) = NumArray(2);
            }
        }

        AlphaArray.deallocate();
        cAlphaFieldNames.deallocate();
        cNumericFieldNames.deallocate();
        NumArray.deallocate();
        lAlphaBlanks.deallocate();
        lNumericBlanks.deallocate();

        if (ErrorsFound) {
            ShowFatalError(state, "GetHeatBalHAMTInput: Errors found getting input.  Program terminates.");
        }
    }

    void InitHeatBalHAMT(EnergyPlusData &state)
    {
        // SUBROUTINE INFORMATION:
        //       AUTHOR         Phillip Biddulph
        //       DATE WRITTEN   June 2008
        //       MODIFIED       B. Griffith, Aug 2012 for surface-specific algorithms
        //       RE-ENGINEERED  na

        // Using/Aliasing
        using General::ScanForReports;

        // Locals
        // SUBROUTINE PARAMETER DEFINITIONS:
        Real64 constexpr adjdist(0.00005); // Allowable distance between two cells, also used as limit on cell length
        static constexpr std::string_view RoutineName("InitCombinedHeatAndMoistureFiniteElement: ");

        // SUBROUTINE LOCAL VARIABLE DECLARATIONS:
        int conid;
        int errorCount;

        Real64 runor;
        Real64 testlen;
        Real64 waterd; // water density
        bool DoReport;

        auto &s_mat = state.dataMaterial;
        auto &s_hbh = state.dataHeatBalHAMTMgr;

        s_hbh->deltat = state.dataGlobal->TimeStepZone * 3600.0;

        // Check the materials information and work out how many cells are required.
        errorCount = 0;
        s_hbh->TotCellsMax = 0;
        for (int sid = 1; sid <= state.dataSurface->TotSurfaces; ++sid) {
            auto const &surf = state.dataSurface->Surface(sid);
            if (surf.Class == SurfaceClass::Window) {
                continue;
            }
            if (surf.HeatTransferAlgorithm != DataSurfaces::HeatTransferModel::HAMT) {
                continue;
            }

            if (surf.Construction == 0) {
                continue;
            }
            auto const &constr = state.dataConstruction->Construct(surf.Construction);

            for (int lid = 1; lid <= constr.TotLayers; ++lid) {
                auto *mat = s_mat->materials(constr.LayerPoint(lid));
                if (mat->ROnly) {
                    ShowSevereError(state, EnergyPlus::format("{}Construction={} cannot contain R-only value materials.", RoutineName, constr.Name));
                    ShowContinueError(state, EnergyPlus::format("Reference Material=\"{}\".", mat->Name));
                    ++errorCount;
                    continue;
                }

                auto *matHAMT = dynamic_cast<MaterialHAMT *>(mat);
                assert(matHAMT != nullptr);

                if (matHAMT->nmu < 0) {
                    ShowSevereError(state, EnergyPlus::format("{}Construction={}", RoutineName, constr.Name));
                    ShowContinueError(
                        state,
                        EnergyPlus::format("Reference Material=\"{}\" does not have required Water Vapor Diffusion Resistance Factor (mu) data.",
                                           matHAMT->Name));
                    ++errorCount;
                }

                if (matHAMT->niso < 0) {
                    ShowSevereError(state, EnergyPlus::format("{}Construction={}", RoutineName, constr.Name));
                    ShowContinueError(state, EnergyPlus::format("Reference Material=\"{}\" does not have required isotherm data.", matHAMT->Name));
                    ++errorCount;
                }
                if (matHAMT->nsuc < 0) {
                    ShowSevereError(state, EnergyPlus::format("{}Construction={}", RoutineName, constr.Name));
                    ShowContinueError(
                        state,
                        EnergyPlus::format("Reference Material=\"{}\" does not have required liquid transport coefficient (suction) data.",
                                           mat->Name));
                    ++errorCount;
                }
                if (matHAMT->nred < 0) {
                    ShowSevereError(state, EnergyPlus::format("{}Construction={}", RoutineName, constr.Name));
                    ShowContinueError(
                        state,
                        EnergyPlus::format("Reference Material=\"{}\" does not have required liquid transport coefficient (redistribution) data.",
                                           mat->Name));
                    ++errorCount;
                }
                if (matHAMT->ntc < 0) {
                    if (mat->Conductivity > 0) {
                        ShowWarningError(state, std::format("{}Construction={}", RoutineName, constr.Name));
                        ShowContinueError(
                            state,
                            std::format("Reference Material=\"{}\" does not have thermal conductivity data. Using fixed value.", matHAMT->Name));
                        matHAMT->ntc = 2;
                        matHAMT->tcwater(1) = 0.0;
                        matHAMT->tcdata(1) = matHAMT->Conductivity;
                        matHAMT->tcwater(2) = matHAMT->isodata(matHAMT->niso);
                        matHAMT->tcdata(2) = matHAMT->Conductivity;
                    } else {
                        ShowSevereError(state, std::format("{}Construction={}", RoutineName, constr.Name));
                        ShowContinueError(state,
                                          std::format("Reference Material=\"{}\" does not have required thermal conductivity data.", matHAMT->Name));
                        ++errorCount;
                    }
                }

                // convert material water content to RH

                waterd = matHAMT->iwater * matHAMT->Density;
                interp(matHAMT->niso, matHAMT->isodata, matHAMT->isorh, waterd, matHAMT->irh);

                // Build the pre-resampled property tables once per material. The
                // same MaterialHAMT instance may be referenced by multiple
                // surfaces; the guard avoids redundant work.
                if (!matHAMT->isoFast.built) {
                    BuildFastTables(*matHAMT);
                }

                // ── Physics-based Fourier meshing (default path for all schemes) ──────
                // Interior cell size:  dxn = sqrt(alpha * dt * C)  with C = spaceDescritConstant
                //   (N1; default 1.0 → cell ≈ one diffusion length / timestep, Fo = 1)
                // Boundary cell target: h   = sqrt(D    * dt * C_b) with C_b = HAMTboundaryC
                //   (N2; default 0.1 → boundary cell finer than interior)
                // Uses dry-state conductivity/density at initialization (moisture state
                // unknown at this point; irh has been set from iwater but cells not placed).
                Real64 const alpha = matHAMT->Conductivity / (matHAMT->Density * matHAMT->SpecHeat);
                Real64 const dxn   = std::sqrt(alpha * s_hbh->deltat * s_hbh->spaceDescritConstant);
                matHAMT->divs      = static_cast<int>(matHAMT->Thickness / dxn);
                // Apply user-provided floor (N3) if set. HAMTdivmin == 0 means "not specified".
                if (s_hbh->HAMTdivmin > 0 && matHAMT->divs < s_hbh->HAMTdivmin) {
                    matHAMT->divs = s_hbh->HAMTdivmin;
                }

                // ── Boundary refinement (N2 = boundary space discretization constant) ──
                // Ensure the cosine first-cell origin ≤ sqrt(D * dt * C_b) for both
                // thermal diffusivity (alpha) and moisture diffusivity (D_phi).
                // D_phi = (delta_a / mu) * p_sat / dwdphi  [m²/s]
                // We take the minimum (strictest) of the two diffusion lengths.
                if (s_hbh->HAMTboundaryC > 0.0 && matHAMT->Thickness > 0.0) {
                    Real64 h_boundary = std::sqrt(alpha * s_hbh->deltat * s_hbh->HAMTboundaryC);

                    bool const has_mu  = (matHAMT->nmu  > 0);
                    bool const has_iso = (matHAMT->niso > 0);
                    if (has_mu && has_iso && matHAMT->Density > 0.0) {
                        // Property evaluation point for the boundary criterion.
                        // We use a fixed mid-range RH rather than the material's
                        // initial RH (matHAMT->irh) because:
                        //   1. The mesh is a one-time decision at init; making it
                        //      depend on initial conditions would tie cell counts
                        //      to user-supplied initial water content, which is
                        //      orthogonal to the physics of how fast moisture
                        //      can diffuse through the material.
                        //   2. Real sorption isotherms are most linear in the mid
                        //      RH band (≈ 0.3–0.7). The ends are often poorly
                        //      sampled (a single ramp-up segment near RH = 0) or
                        //      capillary-regime-dominated near RH = 1.0. Evaluating
                        //      d(w)/d(φ) at RH = 0.5 picks the slope where the
                        //      data is most informative.
                        constexpr Real64 dwdphi_eval_rh = 0.5;

                        Real64 mu_val = 0.0;
                        interp(matHAMT->nmu, matHAMT->murh, matHAMT->mudata,
                               dwdphi_eval_rh, mu_val);
                        if (mu_val > 0.0) {
                            Real64 dwdphi_val = 0.0, water_dummy = 0.0;
                            interp(matHAMT->niso, matHAMT->isorh, matHAMT->isodata,
                                   dwdphi_eval_rh, water_dummy, dwdphi_val);
                            if (dwdphi_val > 0.0) {
                                Real64 const ambp = (state.dataEnvrn->OutBaroPress > 0.0)
                                                        ? state.dataEnvrn->OutBaroPress
                                                        : 101325.0;
                                Real64 const delta_a = WVDC(matHAMT->itemp, ambp);
                                Real64 const p_sat   = RHtoVP(state, 1.0, matHAMT->itemp);
                                Real64 const D_phi   = (delta_a / mu_val) * p_sat / dwdphi_val;
                                if (D_phi > 0.0) {
                                    h_boundary = std::min(
                                        h_boundary,
                                        std::sqrt(D_phi * s_hbh->deltat * s_hbh->HAMTboundaryC));
                                }
                            }
                        }
                    }

                    // Find N_min: L*(1-cos(π/N))/4 ≤ h_boundary  →  N ≥ π / acos(1 - 4h/L)
                    if (h_boundary > 0.0) {
                        Real64 const ratio = 4.0 * h_boundary / matHAMT->Thickness;
                        if (ratio > 0.0 && ratio < 2.0) { // ratio≥2 → any N satisfies
                            int const boundary_divs = static_cast<int>(
                                std::ceil(Constant::Pi / std::acos(std::max(-1.0, 1.0 - ratio))));
                            if (boundary_divs > matHAMT->divs) matHAMT->divs = boundary_divs;
                        }
                    }
                }

                // Apply user-provided ceiling (N4) if set. HAMTdivmax == 0 means "not specified".
                if (s_hbh->HAMTdivmax > 0 && matHAMT->divs > s_hbh->HAMTdivmax) {
                    matHAMT->divs = s_hbh->HAMTdivmax;
                }
                // Always require at least 1 cell.
                if (matHAMT->divs < 1) matHAMT->divs = 1;
                // Check length of cell - reduce number of divisions if necessary
                Real64 const sin_negPIOvr2 = std::sin(-Constant::Pi / 2.0);
                while (true) {
                    testlen = matHAMT->Thickness *
                              ((std::sin(Constant::Pi * (-1.0 / double(matHAMT->divs)) - Constant::Pi / 2.0) / 2.0) - (sin_negPIOvr2 / 2.0));
                    if (testlen > adjdist) {
                        break;
                    }
                    --matHAMT->divs;
                    if (matHAMT->divs < 1) {
                        ShowSevereError(state, std::format("{}Construction={}", RoutineName, constr.Name));
                        ShowContinueError(state, std::format("Reference Material=\"{}\" is too thin.", matHAMT->Name));
                        ++errorCount;
                        break;
                    }
                }
                s_hbh->TotCellsMax += matHAMT->divs;
            }
            s_hbh->TotCellsMax += 7;
        }

        if (errorCount > 0) {
            ShowFatalError(state, "CombinedHeatAndMoistureFiniteElement: Incomplete data to start solution, program terminates.");
        }

        // Make the cells and initialize
        s_hbh->cells.allocate(s_hbh->TotCellsMax);
        for (auto &e : s_hbh->cells) {
            e.adjs = -1;
            e.adjsl = -1;
        }

        if (s_hbh->schemeType != HAMTScheme::GaussSeidel) {
            s_hbh->thomas_a.allocate(s_hbh->TotCellsMax);
            s_hbh->thomas_b.allocate(s_hbh->TotCellsMax);
            s_hbh->thomas_c.allocate(s_hbh->TotCellsMax);
            s_hbh->thomas_d.allocate(s_hbh->TotCellsMax);
            s_hbh->thomas_x.allocate(s_hbh->TotCellsMax);
            s_hbh->thomas_tempp1_prev.allocate(s_hbh->TotCellsMax);
            s_hbh->thomas_rhp1_prev.allocate(s_hbh->TotCellsMax);
        }

        int cid = 0;

        // Set up surface cell structure
        for (int sid = 1; sid <= state.dataSurface->TotSurfaces; ++sid) {
            auto &surf = state.dataSurface->Surface(sid);
            if (!surf.HeatTransSurf) {
                continue;
            }
            if (surf.Class == SurfaceClass::Window) {
                continue;
            }
            if (surf.HeatTransferAlgorithm != DataSurfaces::HeatTransferModel::HAMT) {
                continue;
            }
            // Boundary Cells
            runor = -0.02;
            // Air Convection Cell
            ++cid;
            s_hbh->firstcell(sid) = cid;
            s_hbh->ExtConcell(sid) = cid;
            auto &airConvCell = s_hbh->cells(cid);
            airConvCell.rh = 0.0;
            airConvCell.sid = sid;
            airConvCell.length(1) = 0.01;
            airConvCell.origin(1) = airConvCell.length(1) / 2.0 + runor;

            // Air Radiation Cell
            ++cid;
            s_hbh->ExtRadcell(sid) = cid;
            auto &airRadCell = s_hbh->cells(cid);
            airRadCell.rh = 0.0;
            airRadCell.sid = sid;
            airRadCell.length(1) = 0.01;
            airRadCell.origin(1) = airRadCell.length(1) / 2.0 + runor;

            // Sky Cell
            ++cid;
            s_hbh->ExtSkycell(sid) = cid;
            auto &skyCell = s_hbh->cells(cid);
            skyCell.rh = 0.0;
            skyCell.sid = sid;
            skyCell.length(1) = 0.01;
            skyCell.origin(1) = skyCell.length(1) / 2.0 + runor;

            // Ground Cell
            ++cid;
            s_hbh->ExtGrncell(sid) = cid;
            auto &groundCell = s_hbh->cells(cid);
            groundCell.rh = 0.0;
            groundCell.sid = sid;
            groundCell.length(1) = 0.01;
            groundCell.origin(1) = groundCell.length(1) / 2.0 + runor;
            runor += groundCell.length(1);

            // External Virtual Cell
            ++cid;
            s_hbh->Extcell(sid) = cid;
            auto &extVirtCell = s_hbh->cells(cid);
            extVirtCell.rh = 0.0;
            extVirtCell.sid = sid;
            extVirtCell.length(1) = 0.01;
            extVirtCell.origin(1) = extVirtCell.length(1) / 2.0 + runor;
            runor += extVirtCell.length(1);

            // Material Cells
            auto const &constr = state.dataConstruction->Construct(surf.Construction);
            for (int lid = 1; lid <= constr.TotLayers; ++lid) {
                auto const *mat = dynamic_cast<const MaterialHAMT *>(s_mat->materials(constr.LayerPoint(lid)));
                assert(mat != nullptr);

                for (int did = 1; did <= mat->divs; ++did) {
                    ++cid;

                    auto &matCell = s_hbh->cells(cid);
                    matCell.matid = mat->Num;
                    matCell.mat = mat;  // cache pointer to avoid dynamic_cast in hot loop
                    matCell.sid = sid;

                    matCell.temp = mat->itemp;
                    matCell.tempp1 = mat->itemp;
                    matCell.tempp2 = mat->itemp;

                    matCell.rh = mat->irh;
                    matCell.rhp1 = mat->irh;
                    matCell.rhp2 = mat->irh;

                    matCell.density = mat->Density;
                    matCell.spech = mat->SpecHeat;

                    // Make cells smaller near the surface
                    matCell.length(1) =
                        mat->Thickness * ((std::sin(Constant::Pi * (-double(did) / double(mat->divs)) - Constant::Pi / 2.0) / 2.0) -
                                          (std::sin(Constant::Pi * (-double(did - 1) / double(mat->divs)) - Constant::Pi / 2.0) / 2.0));

                    matCell.origin(1) = runor + matCell.length(1) / 2.0;
                    runor += matCell.length(1);

                    matCell.volume = matCell.length(1) * state.dataSurface->Surface(sid).Area;
                }
            }

            // Interior Virtual Cell
            ++cid;
            s_hbh->Intcell(sid) = cid;
            auto &intVirtCell = s_hbh->cells(cid);
            intVirtCell.sid = sid;
            intVirtCell.rh = 0.0;
            intVirtCell.length(1) = 0.01;
            intVirtCell.origin(1) = intVirtCell.length(1) / 2.0 + runor;
            runor += intVirtCell.length(1);

            // Air Convection Cell
            ++cid;
            s_hbh->lastcell(sid) = cid;
            s_hbh->IntConcell(sid) = cid;
            auto &airConvCell2 = s_hbh->cells(cid);
            airConvCell2.rh = 0.0;
            airConvCell2.sid = sid;
            airConvCell2.length(1) = 0.01;
            airConvCell2.origin(1) = airConvCell2.length(1) / 2.0 + runor;
        }

        // Find adjacent cells.
        for (int cid1 = 1; cid1 <= s_hbh->TotCellsMax; ++cid1) {
            for (int cid2 = 1; cid2 <= s_hbh->TotCellsMax; ++cid2) {
                if (cid1 == cid2) {
                    continue;
                }

                auto &cell1 = s_hbh->cells(cid1);
                auto &cell2 = s_hbh->cells(cid2);

                if (cell1.sid != cell2.sid) {
                    continue;
                }

                Real64 high1 = cell1.origin(1) + cell1.length(1) / 2.0;
                Real64 low2 = cell2.origin(1) - cell2.length(1) / 2.0;
                if (std::abs(low2 - high1) < adjdist) {
                    int adj1 = 0;
                    for (int ii = 1; ii <= adjmax; ++ii) {
                        ++adj1;
                        if (cell1.adjs(adj1) == -1) {
                            break;
                        }
                    }
                    int adj2 = 0;
                    for (int ii = 1; ii <= adjmax; ++ii) {
                        ++adj2;
                        if (cell2.adjs(adj2) == -1) {
                            break;
                        }
                    }
                    cell1.adjs(adj1) = cid2;
                    cell2.adjs(adj2) = cid1;

                    cell1.adjsl(adj1) = adj2;
                    cell2.adjsl(adj2) = adj1;

                    int const surfNum = cell1.sid;
                    cell1.overlap(adj1) = state.dataSurface->Surface(surfNum).Area;
                    cell2.overlap(adj2) = state.dataSurface->Surface(surfNum).Area;
                    cell1.dist(adj1) = cell1.length(1) / 2.0;
                    cell2.dist(adj2) = cell2.length(1) / 2.0;
                }
            }
        }

        // Phase 7: precompute neighbour count per cell. Lets the Picard inner
        // loops use `for (ii in 1..cell.nadj)` and skip the `if (adj == -1) break`
        // sentinel check at every iteration. For 1D walls, nadj is typically
        // 2 for interior cells and 1-5 for boundary cells.
        for (int cid = 1; cid <= s_hbh->TotCellsMax; ++cid) {
            auto &cell = s_hbh->cells(cid);
            int count = 0;
            for (int ii = 1; ii <= adjmax; ++ii) {
                if (cell.adjs(ii) != -1) ++count;
                else break; // adjs are filled densely from the start
            }
            cell.nadj = count;
        }

        // Reset surface virtual cell origins and volumes. Initialize report variables.
        static constexpr std::string_view Format_1966("! <HAMT cells>, Surface Name, Construction Name, Cell Numbers\n");
        print(state.files.eio, Format_1966);
        static constexpr std::string_view Format_1965("! <HAMT origins>, Surface Name, Construction Name, Cell origins (m) \n");
        print(state.files.eio, Format_1965);
        // cCurrentModuleObject='MaterialProperty:HeatAndMoistureTransfer:*'
        for (int sid = 1; sid <= state.dataSurface->TotSurfaces; ++sid) {
            if (!state.dataSurface->Surface(sid).HeatTransSurf) {
                continue;
            }
            if (state.dataSurface->Surface(sid).Class == SurfaceClass::Window) {
                continue;
            }
            if (state.dataSurface->Surface(sid).HeatTransferAlgorithm != DataSurfaces::HeatTransferModel::HAMT) {
                continue;
            }
            s_hbh->cells(s_hbh->Extcell(sid)).origin(1) += s_hbh->cells(s_hbh->Extcell(sid)).length(1) / 2.0;
            s_hbh->cells(s_hbh->Intcell(sid)).origin(1) -= s_hbh->cells(s_hbh->Intcell(sid)).length(1) / 2.0;
            s_hbh->cells(s_hbh->Extcell(sid)).volume = 0.0;
            s_hbh->cells(s_hbh->Intcell(sid)).volume = 0.0;
            s_hbh->watertot(sid) = 0.0;
            s_hbh->surfrh(sid) = 0.0;
            s_hbh->surfextrh(sid) = 0.0;
            s_hbh->surftemp(sid) = 0.0;
            s_hbh->surfexttemp(sid) = 0.0;
            s_hbh->surfvp(sid) = 0.0;
            s_hbh->surfoutvp(sid) = 0.0;
            SetupOutputVariable(state,
                                "HAMT Surface Average Water Content Ratio",
                                Constant::Units::kg_kg,
                                s_hbh->watertot(sid),
                                OutputProcessor::TimeStepType::Zone,
                                OutputProcessor::StoreType::Average,
                                state.dataSurface->Surface(sid).Name);
            SetupOutputVariable(state,
                                "HAMT Surface Inside Face Temperature",
                                Constant::Units::C,
                                s_hbh->surftemp(sid),
                                OutputProcessor::TimeStepType::Zone,
                                OutputProcessor::StoreType::Average,
                                state.dataSurface->Surface(sid).Name);
            SetupOutputVariable(state,
                                "HAMT Surface Inside Face Relative Humidity",
                                Constant::Units::Perc,
                                s_hbh->surfrh(sid),
                                OutputProcessor::TimeStepType::Zone,
                                OutputProcessor::StoreType::Average,
                                state.dataSurface->Surface(sid).Name);
            SetupOutputVariable(state,
                                "HAMT Surface Inside Face Vapor Pressure",
                                Constant::Units::Pa,
                                s_hbh->surfvp(sid),
                                OutputProcessor::TimeStepType::Zone,
                                OutputProcessor::StoreType::Average,
                                state.dataSurface->Surface(sid).Name);
            SetupOutputVariable(state,
                                "HAMT Surface Outside Face Temperature",
                                Constant::Units::C,
                                s_hbh->surfexttemp(sid),
                                OutputProcessor::TimeStepType::Zone,
                                OutputProcessor::StoreType::Average,
                                state.dataSurface->Surface(sid).Name);
            SetupOutputVariable(state,
                                "HAMT Surface Outside Face Relative Humidity",
                                Constant::Units::Perc,
                                s_hbh->surfextrh(sid),
                                OutputProcessor::TimeStepType::Zone,
                                OutputProcessor::StoreType::Average,
                                state.dataSurface->Surface(sid).Name);
            SetupOutputVariable(state,
                                "HAMT Surface Outside Face Vapor Pressure",
                                Constant::Units::Pa,
                                s_hbh->surfoutvp(sid),
                                OutputProcessor::TimeStepType::Zone,
                                OutputProcessor::StoreType::Average,
                                state.dataSurface->Surface(sid).Name);
            // Phase 9 diagnostic: how many iterations the linearized solver did
            // on the most recent call for this surface. 1 = fast path used;
            // 2+ = safety net fired (state changed enough to need property
            // refresh). Useful for tuning the N9/N10 linearization safety
            // thresholds on a per-construction basis.
            SetupOutputVariable(state,
                                "HAMT Surface Linearization Iterations",
                                Constant::Units::None,
                                s_hbh->lastIterCount(sid),
                                OutputProcessor::TimeStepType::Zone,
                                OutputProcessor::StoreType::Average,
                                state.dataSurface->Surface(sid).Name);

            // write cell origins to initialization output file
            conid = state.dataSurface->Surface(sid).Construction;
            print(state.files.eio, "HAMT cells, {},{}", state.dataSurface->Surface(sid).Name, state.dataConstruction->Construct(conid).Name);
            for (int concell = 1, concell_end = s_hbh->Intcell(sid) - s_hbh->Extcell(sid) + 1; concell <= concell_end; ++concell) {
                print(state.files.eio, ",{:4}", concell);
            }
            print(state.files.eio, "\n");
            print(state.files.eio, "HAMT origins,{},{}", state.dataSurface->Surface(sid).Name, state.dataConstruction->Construct(conid).Name);
            for (int cellid = s_hbh->Extcell(sid); cellid <= s_hbh->Intcell(sid); ++cellid) {
                print(state.files.eio, ",{:10.7F}", s_hbh->cells(cellid).origin(1));
            }
            print(state.files.eio, "\n");

            for (int cellid = s_hbh->Extcell(sid), concell = 1; cellid <= s_hbh->Intcell(sid); ++cellid, ++concell) {
                SetupOutputVariable(state,
                                    std::format("HAMT Surface Temperature Cell {}", concell),
                                    Constant::Units::C,
                                    s_hbh->cells(cellid).temp,
                                    OutputProcessor::TimeStepType::Zone,
                                    OutputProcessor::StoreType::Average,
                                    state.dataSurface->Surface(sid).Name);
            }
            for (int cellid = s_hbh->Extcell(sid), concell = 1; cellid <= s_hbh->Intcell(sid); ++cellid, ++concell) {
                SetupOutputVariable(state,
                                    std::format("HAMT Surface Water Content Cell {}", concell),
                                    Constant::Units::kg_kg,
                                    s_hbh->cells(cellid).wreport,
                                    OutputProcessor::TimeStepType::Zone,
                                    OutputProcessor::StoreType::Average,
                                    state.dataSurface->Surface(sid).Name);
            }
            for (int cellid = s_hbh->Extcell(sid), concell = 1; cellid <= s_hbh->Intcell(sid); ++cellid, ++concell) {
                SetupOutputVariable(state,
                                    std::format("HAMT Surface Relative Humidity Cell {}", concell),
                                    Constant::Units::Perc,
                                    s_hbh->cells(cellid).rhp,
                                    OutputProcessor::TimeStepType::Zone,
                                    OutputProcessor::StoreType::Average,
                                    state.dataSurface->Surface(sid).Name);
            }
            for (int cellid = s_hbh->Extcell(sid), concell = 1; cellid <= s_hbh->Intcell(sid); ++cellid, ++concell) {
                SetupOutputVariable(state,
                                    std::format("HAMT Surface Heat Flux Cell {}", concell),
                                    Constant::Units::W_m2,
                                    s_hbh->cells(cellid).qflux,
                                    OutputProcessor::TimeStepType::Zone,
                                    OutputProcessor::StoreType::Average,
                                    state.dataSurface->Surface(sid).Name);
            }
            // Per-cell vapor pressure: completes the thermodynamic state vector alongside
            // Temperature, RH, and Water Content. Required for interstitial condensation
            // analysis (dew-point checks at each spatial node).
            for (int cellid = s_hbh->Extcell(sid), concell = 1; cellid <= s_hbh->Intcell(sid); ++cellid, ++concell) {
                SetupOutputVariable(state,
                                    std::format("HAMT Surface Vapor Pressure Cell {}", concell),
                                    Constant::Units::Pa,
                                    s_hbh->cells(cellid).vpreport,
                                    OutputProcessor::TimeStepType::Zone,
                                    OutputProcessor::StoreType::Average,
                                    state.dataSurface->Surface(sid).Name);
            }
        }

        ScanForReports(state, "Constructions", DoReport, "Constructions");
        if (DoReport) {

            static constexpr std::string_view Format_108("! <Material Nominal Resistance>, Material Name,  Nominal R\n");
            print(state.files.eio, Format_108);

            for (auto const *mat : s_mat->materials) {
                static constexpr std::string_view Format_111("Material Nominal Resistance,{},{:.4R}\n");
                print(state.files.eio, Format_111, mat->Name, mat->NominalR);
            }
        }
    }

    void thomas_solve(Array1D<Real64> &a, Array1D<Real64> &b, Array1D<Real64> const &c, Array1D<Real64> &d, Array1D<Real64> &x, int const N)
    {
        // Forward elimination — removes sub-diagonal in O(N)
        for (int i = 2; i <= N; ++i) {
            Real64 const m = a(i) / b(i - 1);
            b(i) -= m * c(i - 1);
            d(i) -= m * d(i - 1);
        }
        // Back-substitution
        x(N) = d(N) / b(N);
        for (int i = N - 1; i >= 1; --i) {
            x(i) = (d(i) - c(i) * x(i + 1)) / b(i);
        }
    }

    void CalcHeatBalHAMT(EnergyPlusData &state, int const sid, Real64 &SurfTempInTmp, Real64 &TempSurfOutTmp)
    {
        // SUBROUTINE INFORMATION:
        //       AUTHOR         Phillip Biddulph
        //       DATE WRITTEN   June 2008
        //       MODIFIED       na
        //       RE-ENGINEERED  na

        // PURPOSE OF THIS SUBROUTINE:
        // To calculate the heat and moisture transfer through the surface

        // Using/Aliasing
        using DataSurfaces::OtherSideCondModeledExt;

        // Locals
        // SUBROUTINE ARGUMENT DEFINITIONS:

        // SUBROUTINE PARAMETER DEFINITIONS:
        static std::string const HAMTExt("HAMT-Ext");
        static std::string const HAMTInt("HAMT-Int");

        // SUBROUTINE LOCAL VARIABLE DECLARATIONS:
        Real64 SurfTempInP;
        Real64 RhoIn;
        Real64 RhoOut;
        Real64 torsum;
        Real64 oorsum;
        Real64 phioosum;
        Real64 phiorsum;
        Real64 vpoosum;
        Real64 vporsum;
        Real64 rhr1;
        Real64 rhr2;
        Real64 wcap;
        Real64 thermr1;
        Real64 thermr2;
        Real64 tcap;
        Real64 qvp;
        Real64 vaporr1;
        Real64 vaporr2;
        Real64 vpdiff;
        Real64 sumtp1;
        Real64 tempmax;
        Real64 tempmin;

        int itter;

        Real64 denominator;

        auto &s_mat = state.dataMaterial;
        auto &s_hbh = state.dataHeatBalHAMTMgr;

        if (state.dataGlobal->BeginEnvrnFlag && s_hbh->MyEnvrnFlag(sid)) {
            auto &extCell = s_hbh->cells(s_hbh->Extcell(sid));
            extCell.rh = 0.0;
            extCell.rhp1 = 0.0;
            extCell.rhp2 = 0.0;

            extCell.temp = 10.0;
            extCell.tempp1 = 10.0;
            extCell.tempp2 = 10.0;

            auto &intCell = s_hbh->cells(s_hbh->Intcell(sid));
            intCell.rh = 0.0;
            intCell.rhp1 = 0.0;
            intCell.rhp2 = 0.0;

            intCell.temp = 10.0;
            intCell.tempp1 = 10.0;
            intCell.tempp2 = 10.0;

            for (int cid = s_hbh->Extcell(sid) + 1; cid <= s_hbh->Intcell(sid) - 1; ++cid) {
                auto &cell = s_hbh->cells(cid);
                auto const *mat = cell.mat;  // cached at InitHeatBalHAMT
                assert(mat != nullptr);
                cell.temp = mat->itemp;
                cell.tempp1 = mat->itemp;
                cell.tempp2 = mat->itemp;

                cell.rh = mat->irh;
                cell.rhp1 = mat->irh;
                cell.rhp2 = mat->irh;
            }
            // BDF2 history: invalidate so the first step of this environment
            // falls back to backward Euler until UpdateHeatBalHAMT has
            // promoted a real T^n into temp_prev.
            for (int cid = s_hbh->firstcell(sid); cid <= s_hbh->lastcell(sid); ++cid) {
                auto &cell = s_hbh->cells(cid);
                cell.temp_prev_valid = false;
                cell.rh_prev_valid   = false;
            }
            // Phase 8: BC extrapolation history — also invalidated on env restart.
            // First step of the new environment uses raw current BC value;
            // second step uses linear extrapolation; third+ uses minmod-limited.
            s_hbh->bcHistoryDepth(sid) = 0;
            s_hbh->MyEnvrnFlag(sid) = false;
        }
        if (!state.dataGlobal->BeginEnvrnFlag) {
            s_hbh->MyEnvrnFlag(sid) = true;
        }

        auto &extCell = s_hbh->cells(s_hbh->Extcell(sid));
        auto &extRadCell = s_hbh->cells(s_hbh->ExtRadcell(sid));
        auto &extSkyCell = s_hbh->cells(s_hbh->ExtSkycell(sid));
        auto &extGrnCell = s_hbh->cells(s_hbh->ExtGrncell(sid));
        auto &extConCell = s_hbh->cells(s_hbh->ExtConcell(sid));

        // ── Phase 8: 2nd-order BC extrapolation for BDF2 (minmod-limited) ────
        // The implicit BDF2 solve at t^{n+1} needs BC values at t^{n+1}, but
        // EnergyPlus only exposes BC values evaluated within the current Δt —
        // an O(Δt) BC error that caps the global accuracy at O(Δt).
        //
        // Fix: extrapolate from cached prior-step BC using a minmod-limited
        // slope (minmod limiter: Harten 1983; LeVeque 2002 Sec. 6.9 — see module
        // REFERENCES).  With two history levels available, compute:
        //   slope_n   = T^n − T^{n-1}
        //   slope_nm1 = T^{n-1} − T^{n-2}
        //   limited   = minmod(slope_n, slope_nm1)
        //               = 0                          if sign(slope_n) ≠ sign(slope_nm1)
        //               = sign·min(|slope_n|, |slope_nm1|)  otherwise
        //   T_BC^{n+1} ≈ T^n + limited
        // This is the same limiter used in TVD finite-volume schemes for
        // shock-capturing.  It preserves 2nd-order accuracy on smooth segments
        // but zeros out the extrapolation at local extrema (where simple linear
        // extrapolation overshoots).  Falls back to:
        //   depth=2: full minmod limiter
        //   depth=1: simple linear extrapolation 2·T^n − T^{n-1}
        //   depth=0: raw current value (no history yet, env first step)
        // Thomas BE is unaffected (still always uses raw current BC).
        Real64 const tempOutFD_now = state.dataMstBal->TempOutsideAirFD(sid);
        Real64 const spaceMAT_now  = state.dataZoneTempPredictorCorrector
                                         ->spaceHeatBalance(state.dataSurface->Surface(sid).spaceNum).MAT;
        bool const useBDF2 = (s_hbh->schemeType == HAMTScheme::FullyImplicitSecondOrder);
        int  const depth   = s_hbh->bcHistoryDepth(sid);

        auto minmod = [](Real64 a, Real64 b) -> Real64 {
            if (a * b <= 0.0) return 0.0;            // sign flip → no extrapolation
            return (std::abs(a) < std::abs(b)) ? a : b;  // smaller magnitude, same sign
        };
        auto extrap = [&](Real64 now, Real64 prev, Real64 prev2) -> Real64 {
            if (!useBDF2 || depth == 0) return now;       // BE path or no history
            if (depth == 1) return 2.0 * now - prev;      // linear extrap, 1 history
            // depth >= 2: minmod-limited
            Real64 const slope_n   = now  - prev;
            Real64 const slope_nm1 = prev - prev2;
            return now + minmod(slope_n, slope_nm1);
        };
        Real64 const tempOutBC = extrap(tempOutFD_now,
                                        s_hbh->tempOutPrev(sid),
                                        s_hbh->tempOutPrev2(sid));
        Real64 const spaceMAT  = extrap(spaceMAT_now,
                                        s_hbh->spaceMATPrev(sid),
                                        s_hbh->spaceMATPrev2(sid));

        // Set all the boundary values (using extrapolated BC for BDF2; raw for Thomas BE).
        extRadCell.temp = tempOutBC;
        extConCell.temp = tempOutBC;
        if (state.dataSurface->Surface(sid).ExtBoundCond == OtherSideCondModeledExt) {
            // CR8046 switch modeled rad temp for sky temp.
            extSkyCell.temp = state.dataSurface->OSCM(state.dataSurface->Surface(sid).OSCMPtr).TRad;
            extCell.Qadds = 0.0; // eliminate incident shortwave on underlying surface
        } else {
            extSkyCell.temp = state.dataEnvrn->SkyTemp;
            // Exterior heat source: absorbed shortwave plus long-wave radiation exchange
            // with surrounding surfaces (e.g. adjacent buildings defined via
            // SurroundingProperty:SurroundingSurfaces). SurfQRadLWOutSrdSurfs is
            // computed by HeatBalanceSurfaceManager and is zero when no surrounding
            // surfaces are defined, so no guard is required. This brings HAMT into
            // parity with the CTF and CondFD exterior heat balance (GitHub issue #11318).
            extCell.Qadds = state.dataSurface->Surface(sid).Area *
                            (state.dataHeatBalSurf->SurfOpaqQRadSWOutAbs(sid) +
                             state.dataHeatBalSurf->SurfQRadLWOutSrdSurfs(sid));
        }

        extGrnCell.temp = tempOutBC;   // Phase 8: 2nd-order extrapolated BC for BDF2
        RhoOut = state.dataMstBal->RhoVaporAirOut(sid);

        // Special case when the surface is an internal mass
        if (state.dataSurface->Surface(sid).ExtBoundCond == sid) {
            extConCell.temp = spaceMAT;
            RhoOut = state.dataMstBal->RhoVaporAirIn(sid);
        }

        RhoIn = state.dataMstBal->RhoVaporAirIn(sid);

        extRadCell.htc = state.dataMstBal->HAirFD(sid);
        extConCell.htc = state.dataMstBal->HConvExtFD(sid);
        extSkyCell.htc = state.dataMstBal->HSkyFD(sid);
        extGrnCell.htc = state.dataMstBal->HGrndFD(sid);

        auto &intCell = s_hbh->cells(s_hbh->Intcell(sid));
        auto &intConCell = s_hbh->cells(s_hbh->IntConcell(sid));

        intConCell.temp = spaceMAT;
        intConCell.htc = state.dataMstBal->HConvInFD(sid);

        intCell.Qadds = state.dataSurface->Surface(sid).Area *
                        (state.dataHeatBalSurf->SurfOpaqQRadSWInAbs(sid) + state.dataHeatBalSurf->SurfQdotRadNetLWInPerArea(sid) +
                         state.dataHeatBalSurf->SurfQdotRadHVACInPerArea(sid) + state.dataHeatBal->SurfQdotRadIntGainsInPerArea(sid) +
                         state.dataHeatBalSurf->SurfQAdditionalHeatSourceInside(sid));

        extConCell.rh = PsyRhFnTdbRhov(state, extConCell.temp, RhoOut, HAMTExt);
        intConCell.rh = PsyRhFnTdbRhov(state, intConCell.temp, RhoIn, HAMTInt);

        if (extConCell.rh > rhmax) {
            extConCell.rh = rhmax;
        }
        if (intConCell.rh > rhmax) {
            intConCell.rh = rhmax;
        }

        // PDB August 2009 Start! Correction for when no vapour transfer coefficient have been defined.
        if (s_hbh->extvtcflag(sid)) {
            extConCell.vtc = s_hbh->extvtc(sid);
        } else {
            if (extConCell.rh > 0) {
                extConCell.vtc =
                    state.dataMstBal->HMassConvExtFD(sid) * RhoOut / (PsyPsatFnTemp(state, state.dataMstBal->TempOutsideAirFD(sid)) * extConCell.rh);
            } else {
                extConCell.vtc = 10000.0;
            }
        }

        if (s_hbh->intvtcflag(sid)) {
            intConCell.vtc = s_hbh->intvtc(sid);
            state.dataMstBal->HMassConvInFD(sid) = intConCell.vtc * PsyPsatFnTemp(state, spaceMAT) * intConCell.rh / RhoIn;
        } else {
            if (intConCell.rh > 0) {
                intConCell.vtc = state.dataMstBal->HMassConvInFD(sid) * RhoIn / (PsyPsatFnTemp(state, spaceMAT) * intConCell.rh);
            } else {
                intConCell.vtc = 10000.0;
            }
        }
        // PDB August 2009 End

        // Initialise
        for (int cid = s_hbh->firstcell(sid); cid <= s_hbh->Extcell(sid) - 1; ++cid) {
            auto &cell = s_hbh->cells(cid);
            cell.tempp1 = cell.temp;
            cell.tempp2 = cell.temp;
            cell.rhp1 = cell.rh;
            cell.rhp2 = cell.rh;
        }
        for (int cid = s_hbh->Intcell(sid) + 1; cid <= s_hbh->lastcell(sid); ++cid) {
            auto &cell = s_hbh->cells(cid);
            cell.tempp1 = cell.temp;
            cell.tempp2 = cell.temp;
            cell.rhp1 = cell.rh;
            cell.rhp2 = cell.rh;
        }

        if (s_hbh->schemeType == HAMTScheme::GaussSeidel) {
        // ── LEGACY GAUSS-SEIDEL PATH — code below is unchanged ─────────────────
        itter = 0;
        while (true) {
            ++itter;
            // Update Moisture values

            for (int cid = s_hbh->firstcell(sid); cid <= s_hbh->lastcell(sid); ++cid) {
                auto &cell = s_hbh->cells(cid);
                cell.vp = RHtoVP(state, cell.rh, cell.temp);
                cell.vpp1 = RHtoVP(state, cell.rhp1, cell.tempp1);
                cell.vpsat = PsyPsatFnTemp(state, cell.tempp1);
                if (cell.matid > 0) {
                    auto const *mat = cell.mat;  // cached at InitHeatBalHAMT
                    assert(mat != nullptr);
                    // Fast O(1) uniform-grid lookups; the FastTable objects were
                    // built once in InitHeatBalHAMT from the raw isorh/isodata
                    // arrays. Falls back to the legacy linear-search interp() if
                    // the fast table wasn't built (defensive — shouldn't happen
                    // for HAMT-tagged constructions).
                    if (mat->isoFast.built) {
                        fast_interp(mat->isoFast, cell.rhp1, cell.water, &cell.dwdphi);
                    } else {
                        interp(mat->niso, mat->isorh, mat->isodata, cell.rhp1, cell.water, cell.dwdphi);
                    }
                    if (state.dataEnvrn->IsRain && s_hbh->rainswitch) {
                        if (mat->sucFast.built) fast_interp(mat->sucFast, cell.water, cell.dw);
                        else                    interp(mat->nsuc, mat->sucwater, mat->sucdata, cell.water, cell.dw);
                    } else {
                        if (mat->redFast.built) fast_interp(mat->redFast, cell.water, cell.dw);
                        else                    interp(mat->nred, mat->redwater, mat->reddata, cell.water, cell.dw);
                    }
                    if (mat->muFast.built) fast_interp(mat->muFast, cell.rhp1, cell.mu);
                    else                   interp(mat->nmu, mat->murh, mat->mudata, cell.rhp1, cell.mu);
                    if (mat->tcFast.built) fast_interp(mat->tcFast, cell.water, cell.wthermalc);
                    else                   interp(mat->ntc, mat->tcwater, mat->tcdata, cell.water, cell.wthermalc);
                }
            }

            // Calculate Heat and Vapor resistances,
            for (int cid = s_hbh->Extcell(sid); cid <= s_hbh->Intcell(sid); ++cid) {
                torsum = 0.0;
                oorsum = 0.0;
                vpdiff = 0.0;
                auto &cell = s_hbh->cells(cid);
                for (int ii = 1; ii <= adjmax; ++ii) {
                    int adj = cell.adjs(ii);
                    int adjl = cell.adjsl(ii);
                    if (adj == -1) {
                        break;
                    }

                    if (cell.htc > 0) {
                        thermr1 = 1.0 / (cell.overlap(ii) * cell.htc);
                    } else if (cell.matid > 0) {
                        thermr1 = cell.dist(ii) / (cell.overlap(ii) * cell.wthermalc);
                    } else {
                        thermr1 = 0.0;
                    }

                    if (cell.vtc > 0) {
                        vaporr1 = 1.0 / (cell.overlap(ii) * cell.vtc);
                    } else if (cell.matid > 0) {
                        vaporr1 = (cell.dist(ii) * cell.mu) / (cell.overlap(ii) * WVDC(cell.tempp1, state.dataEnvrn->OutBaroPress));
                    } else {
                        vaporr1 = 0.0;
                    }

                    auto &adjCell = s_hbh->cells(adj);
                    if (adjCell.htc > 0) {
                        thermr2 = 1.0 / (cell.overlap(ii) * adjCell.htc);
                    } else if (adjCell.matid > 0) {
                        thermr2 = adjCell.dist(adjl) / (cell.overlap(ii) * adjCell.wthermalc);
                    } else {
                        thermr2 = 0.0;
                    }

                    if (adjCell.vtc > 0) {
                        vaporr2 = 1.0 / (cell.overlap(ii) * adjCell.vtc);
                    } else if (adjCell.matid > 0) {
                        vaporr2 = adjCell.mu * adjCell.dist(adjl) / (WVDC(adjCell.tempp1, state.dataEnvrn->OutBaroPress) * cell.overlap(ii));
                    } else {
                        vaporr2 = 0.0;
                    }

                    if (thermr1 + thermr2 > 0) {
                        oorsum += 1.0 / (thermr1 + thermr2);
                        torsum += adjCell.tempp1 / (thermr1 + thermr2);
                    }
                    if (vaporr1 + vaporr2 > 0) {
                        vpdiff += (adjCell.vp - cell.vp) / (vaporr1 + vaporr2);
                    }
                }

                // Calculate Heat Capacitance
                tcap = ((cell.density * cell.spech + cell.water * wspech) * cell.volume);

                // calculate the latent heat if wanted and check for divergence
                qvp = 0.0;
                if ((cell.matid > 0) && (s_hbh->latswitch)) {
                    qvp = vpdiff * whv;
                }
                if (std::abs(qvp) > qvplim) {
                    if (!state.dataGlobal->WarmupFlag) {
                        ++s_hbh->qvpErrCount;
                        if (s_hbh->qvpErrCount < 16) {
                            ShowWarningError(
                                state,
                                std::format("HeatAndMoistureTransfer: Large Latent Heat for Surface {}", state.dataSurface->Surface(sid).Name));
                        } else {
                            ShowRecurringWarningErrorAtEnd(state, "HeatAndMoistureTransfer: Large Latent Heat Errors ", s_hbh->qvpErrReport);
                        }
                    }
                    qvp = 0.0;
                }

                // Calculate the temperature for the next time step
                cell.tempp1 = (torsum + qvp + cell.Qadds + (tcap * cell.temp / s_hbh->deltat)) / (oorsum + (tcap / s_hbh->deltat));
            }

            // Check for silly temperatures
            tempmax = maxval(s_hbh->cells, &subcell::tempp1);
            tempmin = minval(s_hbh->cells, &subcell::tempp1);
            if (tempmax > state.dataHeatBalSurf->MaxSurfaceTempLimit) {
                if (!state.dataGlobal->WarmupFlag) {
                    if (state.dataSurface->SurfHighTempErrCount(sid) == 0) {
                        ShowSevereMessage(state,
                                          EnergyPlus::format("HAMT: Temperature (high) out of bounds ({:.2R}) for surface={}",
                                                             tempmax,
                                                             state.dataSurface->Surface(sid).Name));
                        ShowContinueErrorTimeStamp(state, "");
                    }
                    ShowRecurringWarningErrorAtEnd(state,
                                                   "HAMT: Temperature Temperature (high) out of bounds; Surface=" +
                                                       state.dataSurface->Surface(sid).Name,
                                                   state.dataSurface->SurfHighTempErrCount(sid),
                                                   tempmax,
                                                   tempmax,
                                                   _,
                                                   "C",
                                                   "C");
                }
            }
            if (tempmax > state.dataHeatBalSurf->MaxSurfaceTempLimitBeforeFatal) {
                if (!state.dataGlobal->WarmupFlag) {
                    ShowSevereError(state,
                                    EnergyPlus::format("HAMT: HAMT: Temperature (high) out of bounds ( {:.2R}) for surface={}",
                                                       tempmax,
                                                       state.dataSurface->Surface(sid).Name));
                    ShowContinueErrorTimeStamp(state, "");
                    ShowFatalError(state, "Program terminates due to preceding condition.");
                }
            }
            if (tempmin < MinSurfaceTempLimit) {
                if (!state.dataGlobal->WarmupFlag) {
                    if (state.dataSurface->SurfHighTempErrCount(sid) == 0) {
                        ShowSevereMessage(state,
                                          EnergyPlus::format("HAMT: Temperature (low) out of bounds ({:.2R}) for surface={}",
                                                             tempmin,
                                                             state.dataSurface->Surface(sid).Name));
                        ShowContinueErrorTimeStamp(state, "");
                    }
                    ShowRecurringWarningErrorAtEnd(state,
                                                   "HAMT: Temperature Temperature (high) out of bounds; Surface=" +
                                                       state.dataSurface->Surface(sid).Name,
                                                   state.dataSurface->SurfHighTempErrCount(sid),
                                                   tempmin,
                                                   tempmin,
                                                   _,
                                                   "C",
                                                   "C");
                }
            }
            if (tempmin < MinSurfaceTempLimitBeforeFatal) {
                if (!state.dataGlobal->WarmupFlag) {
                    ShowSevereError(state,
                                    EnergyPlus::format("HAMT: HAMT: Temperature (low) out of bounds ( {:.2R}) for surface={}",
                                                       tempmin,
                                                       state.dataSurface->Surface(sid).Name));
                    ShowContinueErrorTimeStamp(state, "");
                    ShowFatalError(state, "Program terminates due to preceding condition.");
                }
            }

            // Calculate the liquid and vapor resisitances
            for (int cid = s_hbh->Extcell(sid); cid <= s_hbh->Intcell(sid); ++cid) {
                phioosum = 0.0;
                phiorsum = 0.0;
                vpoosum = 0.0;
                vporsum = 0.0;

                auto &cell = s_hbh->cells(cid);
                for (int ii = 1; ii <= adjmax; ++ii) {
                    int adj = cell.adjs(ii);
                    int adjl = cell.adjsl(ii);
                    if (adj == -1) {
                        break;
                    }

                    if (cell.vtc > 0) {
                        vaporr1 = 1.0 / (cell.overlap(ii) * cell.vtc);
                    } else if (cell.matid > 0) {
                        vaporr1 = (cell.dist(ii) * cell.mu) / (cell.overlap(ii) * WVDC(cell.tempp1, state.dataEnvrn->OutBaroPress));
                    } else {
                        vaporr1 = 0.0;
                    }

                    auto &adjCell = s_hbh->cells(adj);
                    if (adjCell.vtc > 0) {
                        vaporr2 = 1.0 / (cell.overlap(ii) * adjCell.vtc);
                    } else if (adjCell.matid > 0) {
                        vaporr2 = (adjCell.dist(adjl) * adjCell.mu) / (cell.overlap(ii) * WVDC(adjCell.tempp1, state.dataEnvrn->OutBaroPress));
                    } else {
                        vaporr2 = 0.0;
                    }
                    if (vaporr1 + vaporr2 > 0) {
                        vpoosum += 1.0 / (vaporr1 + vaporr2);
                        vporsum += (adjCell.vpp1 / (vaporr1 + vaporr2));
                    }

                    if ((cell.dw > 0) && (cell.dwdphi > 0)) {
                        rhr1 = cell.dist(ii) / (cell.overlap(ii) * cell.dw * cell.dwdphi);
                    } else {
                        rhr1 = 0.0;
                    }
                    if ((adjCell.dw > 0) && (adjCell.dwdphi > 0)) {
                        rhr2 = adjCell.dist(adjl) / (cell.overlap(ii) * adjCell.dw * adjCell.dwdphi);
                    } else {
                        rhr2 = 0.0;
                    }

                    //             IF(rhr1+rhr2>0)THEN
                    if (rhr1 * rhr2 > 0) {
                        phioosum += 1.0 / (rhr1 + rhr2);
                        phiorsum += (adjCell.rhp1 / (rhr1 + rhr2));
                    }
                }

                // Moisture Capacitance
                if (cell.dwdphi > 0.0) {
                    wcap = cell.dwdphi * cell.volume;
                } else {
                    wcap = 0.0;
                }

                // Calculate the RH for the next time step
                denominator = (phioosum + vpoosum * cell.vpsat + wcap / s_hbh->deltat);
                if (denominator != 0.0) {
                    cell.rhp1 = (phiorsum + vporsum + (wcap * cell.rh) / s_hbh->deltat) / denominator;
                } else {
                    ShowSevereError(state, "CalcHeatBalHAMT: denominator in calculating RH is zero.  Check material properties for accuracy.");
                    ShowContinueError(state, std::format("...Problem occurs in Material=\"{}\".", s_mat->materials(cell.matid)->Name));
                    ShowFatalError(state, "Program terminates due to preceding condition.");
                }

                if (cell.rhp1 > rhmax) {
                    cell.rhp1 = rhmax;
                }
            }

            // Check for convergence or too many iterations
            sumtp1 = 0.0;
            for (int cid = s_hbh->Extcell(sid); cid <= s_hbh->Intcell(sid); ++cid) {
                auto const &cell = s_hbh->cells(cid);
                if (sumtp1 < std::abs(cell.tempp2 - cell.tempp1)) {
                    sumtp1 = std::abs(cell.tempp2 - cell.tempp1);
                }
            }
            if (sumtp1 < s_hbh->HAMTconvt) {
                break;
            }
            if (itter > s_hbh->HAMTittermax) {
                // Iteration cap reached — the Gauss-Seidel sweep did not converge
                // within the allowed limit. Issue a recurring warning so the user
                // is informed and knows they can raise N5 (Maximum Iterations) in
                // HeatBalanceSettings:HeatAndMoistureTransfer if needed.
                if (!state.dataGlobal->WarmupFlag) {
                    if (s_hbh->gsIterCapErrCount < 16) {
                        ++s_hbh->gsIterCapErrCount;
                        ShowWarningError(state,
                            std::format("HeatAndMoistureTransfer: HAMT GaussSeidel solver reached the maximum "
                                        "iteration limit ({}) for surface \"{}\"; solution may not be fully converged.",
                                        s_hbh->HAMTittermax, state.dataSurface->Surface(sid).Name));
                        ShowContinueErrorTimeStamp(state, "");
                        ShowContinueError(state,
                            "If this warning recurs, increase 'Maximum Iterations' (N5) in "
                            "HeatBalanceSettings:HeatAndMoistureTransfer, or reduce the simulation "
                            "timestep to improve per-step convergence.");
                    } else {
                        ShowRecurringWarningErrorAtEnd(state,
                            "HeatAndMoistureTransfer: HAMT GaussSeidel solver reached maximum iteration limit.",
                            s_hbh->gsIterCapErrReport);
                    }
                }
                break;
            }
            for (int cid = s_hbh->firstcell(sid); cid <= s_hbh->lastcell(sid); ++cid) {
                auto &cell = s_hbh->cells(cid);
                cell.tempp2 = cell.tempp1;
                cell.rhp2 = cell.rhp1;
            }
        }
        // Record GS iteration count for the "HAMT Surface Linearization Iterations" output
        // variable (shared with the Thomas path — here itter is the last-sweep counter).
        s_hbh->lastIterCount(sid) = static_cast<Real64>(itter);
        // ── END GAUSS-SEIDEL PATH ───────────────────────────────────────────────
        } else {
            CalcHeatBalHAMT_Thomas(state, sid);
        }

        // report back to CalcHeatBalanceInsideSurf
        TempSurfOutTmp = extCell.tempp1;
        SurfTempInTmp = intCell.tempp1;

        SurfTempInP = intCell.rhp1 * PsyPsatFnTemp(state, intCell.tempp1);

        state.dataMstBal->RhoVaporSurfIn(sid) = SurfTempInP / (461.52 * (spaceMAT + Constant::Kelvin));
    }

    void CalcHeatBalHAMT_Thomas(EnergyPlusData &state, int const sid)
    {
        // Thomas (TDMA) solver for the FullyImplicitFirstOrder-Thomas (backward-Euler) and
        // FullyImplicitSecondOrder-Thomas (BDF2 - second-order Backward Differentiation Formula) schemes.
        // Boundary cell temperatures and RH (extConCell, intConCell, etc.) are pre-set by the
        // caller (CalcHeatBalHAMT). This function writes tempp1/rhp1 for Extcell..Intcell.

        auto &s_hbh = state.dataHeatBalHAMTMgr;
        auto &s_mat = state.dataMaterial;

        int const extIdx = s_hbh->Extcell(sid);
        int const intIdx = s_hbh->Intcell(sid);
        int const N = intIdx - extIdx + 1; // cells in solve domain
        int const offset = extIdx - 1;     // maps local 1..N to global cell index

        assert(N <= s_hbh->TotCellsMax);

        // ── Capture entering state for safety check (Phase 3b) ───────────────
        // After iter 1, we compare the solved tempp1/rhp1 against these values.
        // If the change is small (< safety thresholds), the linearization
        // assumption holds and we accept iter 1's solution. Otherwise we do
        // iter 2 with refreshed properties — this protects materials with
        // nonlinear iso/mu curves and rapid wetting events.
        auto &tempp1_entering = s_hbh->thomas_tempp1_prev;
        auto &rhp1_entering   = s_hbh->thomas_rhp1_prev;
        for (int i = 1; i <= N; ++i) {
            tempp1_entering(i) = s_hbh->cells(i + offset).tempp1;
            rhp1_entering(i)   = s_hbh->cells(i + offset).rhp1;
        }

        // ── One-time setup: BC cell properties are constant throughout the call ──
        // cell.vp uses OLD time-level (rh, temp) which never change inside this
        // function. For BC cells (matid <= 0), tempp1/rhp1 are also fixed for
        // the entire CalcHeatBalHAMT call (BC values were set by the caller
        // before entering this function), so their vpp1/vpsat/wvdc are constants.
        // Solve cell properties are updated once below in the property-update
        // block (Phase 3 linearization: one Picard iter, properties frozen).
        Real64 const baroPressOnce = state.dataEnvrn->OutBaroPress;
        for (int cid = s_hbh->firstcell(sid); cid <= s_hbh->lastcell(sid); ++cid) {
            auto &cell = s_hbh->cells(cid);
            cell.vp = RHtoVP(state, cell.rh, cell.temp);
            if (cell.matid <= 0) {
                // BC cell: tempp1/rhp1 are fixed; vpp1/vpsat/wvdc are constant in Picard.
                cell.vpp1  = RHtoVP(state, cell.rhp1, cell.tempp1);
                cell.vpsat = PsyPsatFnTemp(state, cell.tempp1);
                cell.wvdc  = WVDC(cell.tempp1, baroPressOnce);
            }
        }

        // ── Phase 3+3b+4: linearized solve with safety-net iter 2+ ────────────
        // Iter 1 is the fast linearized path: properties evaluated at entering
        // state, assemble + solve heat + moisture. For most timesteps this
        // single iteration is sufficient — property changes between iterates
        // are below FP precision for typical (smooth-iso) materials.
        //
        // After iter 1, check the state-change magnitude against
        // linearizationSafetyTemp / linearizationSafetyRH. If under threshold,
        // we accept iter 1. Otherwise, fall through to iter 2 with refreshed
        // properties at iter-1's solved state. Iter 2+ uses the standard
        // convergence threshold (HAMTconvt / HAMTconvphi) — so the algorithm
        // gracefully degrades to original Picard-with-refresh behaviour
        // whenever the linearization assumption breaks.
        //
        // For 1D CLT in Denver weather, iter 2 fires < 0.05% of the time.
        // For materials with sharp nonlinearity or rapid transients, iter 2+
        // fires more often — exactly when needed. Max iters is bounded by
        // HAMTittermax (IDD field N4).
        Real64 const safetyT   = s_hbh->linearizationSafetyTemp;
        Real64 const safetyPhi = s_hbh->linearizationSafetyRH;
        Real64 const convT     = s_hbh->HAMTconvt;
        Real64 const convPhi   = s_hbh->HAMTconvphi;
        int const maxIter      = std::max(2, s_hbh->HAMTittermax);

        // tempp1_entering / rhp1_entering currently hold the entering state
        // (captured above for the safety check after iter 1). For iter 2+ we
        // need to track the previous-iter state instead — we'll overwrite the
        // same buffer between iters.

        int nIters = 0;
        bool converged = false; // set to true at every converging break below
        for (int outer = 1; outer <= maxIter; ++outer) {
            ++nIters;

        // ── Property update — solve domain only (BC cells handled above) ─────
        {
            for (int cid = extIdx; cid <= intIdx; ++cid) {
                auto &cell = s_hbh->cells(cid);
                cell.vpp1  = RHtoVP(state, cell.rhp1, cell.tempp1);
                cell.vpsat = PsyPsatFnTemp(state, cell.tempp1);
                cell.wvdc  = WVDC(cell.tempp1, baroPressOnce);
                if (cell.matid > 0) {
                    auto const *mat = cell.mat;  // cached at InitHeatBalHAMT
                    assert(mat != nullptr);
                    // Fast O(1) uniform-grid lookups; the FastTable objects were
                    // built once in InitHeatBalHAMT from the raw isorh/isodata
                    // arrays. Falls back to the legacy linear-search interp() if
                    // the fast table wasn't built (defensive — shouldn't happen
                    // for HAMT-tagged constructions).
                    if (mat->isoFast.built) {
                        fast_interp(mat->isoFast, cell.rhp1, cell.water, &cell.dwdphi);
                    } else {
                        interp(mat->niso, mat->isorh, mat->isodata, cell.rhp1, cell.water, cell.dwdphi);
                    }
                    if (state.dataEnvrn->IsRain && s_hbh->rainswitch) {
                        if (mat->sucFast.built) fast_interp(mat->sucFast, cell.water, cell.dw);
                        else                    interp(mat->nsuc, mat->sucwater, mat->sucdata, cell.water, cell.dw);
                    } else {
                        if (mat->redFast.built) fast_interp(mat->redFast, cell.water, cell.dw);
                        else                    interp(mat->nred, mat->redwater, mat->reddata, cell.water, cell.dw);
                    }
                    if (mat->muFast.built) fast_interp(mat->muFast, cell.rhp1, cell.mu);
                    else                   interp(mat->nmu, mat->murh, mat->mudata, cell.rhp1, cell.mu);
                    if (mat->tcFast.built) fast_interp(mat->tcFast, cell.water, cell.wthermalc);
                    else                   interp(mat->ntc, mat->tcwater, mat->tcdata, cell.water, cell.wthermalc);
                }
            }
        }

            // ── Assemble heat tridiagonal ────────────────────────────────────────
            // Fused neighbor walk: latent-heat vapor diffusion (qvp) and thermal
            // conductance (G) used to be computed in two separate passes over the
            // same neighbor list. Both use the same cell.tempp1 (current Picard
            // iteration value), so we walk neighbors once and compute both.
            bool const doLatent = s_hbh->latswitch;
            Real64 const baroPress = state.dataEnvrn->OutBaroPress;
            // Second-order BDF2 in time uses the previous timestep's converged
            // result (cell.temp_prev) in addition to cell.temp (= T^n). The
            // first step of each environment falls back to backward Euler
            // because no T^{n-1} history exists yet — that's gated per cell
            // by cell.temp_prev_valid (set true in UpdateHeatBalHAMT after
            // the first successful step).
            bool const useBDF2 = (s_hbh->schemeType == HAMTScheme::FullyImplicitSecondOrder);
            for (int i = 1; i <= N; ++i) {
                int const cid = i + offset;
                auto &cell = s_hbh->cells(cid);

                // Heat storage term (Phase 7: hoist tcap/dt and outer-cell-only
                // property reciprocals out of the inner loop)
                Real64 const tcap_over_dt = (cell.density * cell.spech + cell.water * wspech) * cell.volume / s_hbh->deltat;
                Real64 const inv_wthermalc = (cell.wthermalc > 0.0) ? (1.0 / cell.wthermalc) : 0.0;
                Real64 const mu_over_wvdc  = (cell.wvdc > 0.0) ? (cell.mu / cell.wvdc) : 0.0;

                // Time-discretisation coefficients (see module REFERENCES:
                // Ascher & Petzold 1998, Sec. 5.1-5.2 for the BDF2 weights):
                //   Backward Euler:   diag = M/dt,       rhs = (M/dt)·T^n
                //   BDF2:             diag = 1.5·M/dt,   rhs = (M/dt)·(2·T^n − 0.5·T^{n−1})
                // The 2nd-order BDF2 weights (1.5, -2, 0.5) are A- and L-stable.
                // Both reduce to the same expression when temp_prev == temp, which is
                // why the per-cell fallback to backward Euler on the first timestep
                // (no T^{n-1} history yet) is exact rather than approximate.
                Real64 b_mass, d_history_T;
                if (useBDF2 && cell.temp_prev_valid) {
                    b_mass       = 1.5 * tcap_over_dt;
                    d_history_T  = tcap_over_dt * (2.0 * cell.temp - 0.5 * cell.temp_prev);
                } else {
                    b_mass       = tcap_over_dt;
                    d_history_T  = tcap_over_dt * cell.temp;
                }

                s_hbh->thomas_a(i) = 0.0;
                s_hbh->thomas_b(i) = b_mass;
                s_hbh->thomas_c(i) = 0.0;
                s_hbh->thomas_d(i) = d_history_T + cell.Qadds;

                // The outer cell is always a solve cell (matid > 0, htc/vtc <= 0
                // by construction in InitHeatBalHAMT). So the "is BC cell?"
                // branches below collapse to the material case directly.
                bool const haveLatent = doLatent && (cell.matid > 0);
                Real64 vpdiff = 0.0;

                for (int ii = 1; ii <= adjmax; ++ii) {
                    int const adj = cell.adjs(ii);
                    if (adj == -1) break;
                    int const adjl = cell.adjsl(ii);
                    auto &adjCell = s_hbh->cells(adj);
                    Real64 const ovl = cell.overlap(ii);
                    Real64 const inv_ovl = 1.0 / ovl;

                    // --- Thermal conductance ---
                    // NB: the "solve domain" [Extcell, Intcell] actually starts and
                    // ends with VIRTUAL cells (matid = -1, wthermalc = 0). Their
                    // contribution is zero, and the outer-cell branches must remain
                    // to handle them — we cannot assume cell.matid > 0 even for
                    // cells in the solve domain. The Phase 7 hoist (inv_wthermalc)
                    // is used inside the matid>0 branch where it applies.
                    Real64 thermr1, thermr2;
                    if (cell.htc > 0) {
                        thermr1 = inv_ovl / cell.htc;
                    } else if (cell.matid > 0) {
                        thermr1 = cell.dist(ii) * inv_wthermalc * inv_ovl;
                    } else {
                        thermr1 = 0.0;
                    }
                    if (adjCell.htc > 0) {
                        thermr2 = inv_ovl / adjCell.htc;
                    } else if (adjCell.matid > 0) {
                        thermr2 = adjCell.dist(adjl) * inv_ovl / adjCell.wthermalc;
                    } else {
                        thermr2 = 0.0;
                    }

                    if (thermr1 + thermr2 > 0) {
                        Real64 const G = 1.0 / (thermr1 + thermr2);
                        s_hbh->thomas_b(i) += G;
                        int const local_j = adj - offset;
                        if (local_j >= 1 && local_j <= N) {
                            if (local_j == i - 1) s_hbh->thomas_a(i) = -G;
                            if (local_j == i + 1) s_hbh->thomas_c(i) = -G;
                        } else {
                            s_hbh->thomas_d(i) += G * adjCell.tempp1; // BC absorption
                        }
                    }

                    // --- Latent-heat vapor diffusion (only when enabled) ---
                    if (haveLatent) {
                        Real64 vaporr1, vaporr2;
                        if (cell.vtc > 0) {
                            vaporr1 = inv_ovl / cell.vtc;
                        } else if (cell.matid > 0) {
                            vaporr1 = cell.dist(ii) * mu_over_wvdc * inv_ovl;
                        } else {
                            vaporr1 = 0.0;
                        }
                        if (adjCell.vtc > 0) {
                            vaporr2 = inv_ovl / adjCell.vtc;
                        } else if (adjCell.matid > 0) {
                            vaporr2 = adjCell.dist(adjl) * adjCell.mu * inv_ovl / adjCell.wvdc;
                        } else {
                            vaporr2 = 0.0;
                        }
                        if (vaporr1 + vaporr2 > 0) {
                            vpdiff += (adjCell.vp - cell.vp) / (vaporr1 + vaporr2);
                        }
                    }
                }

                // Latent-heat source term
                if (haveLatent) {
                    Real64 qvp = vpdiff * whv;
                    if (std::abs(qvp) > qvplim) {
                        if (!state.dataGlobal->WarmupFlag) {
                            ++s_hbh->qvpErrCount;
                            if (s_hbh->qvpErrCount < 16) {
                                ShowWarningError(state,
                                    std::format("HeatAndMoistureTransfer: Large Latent Heat for Surface {}",
                                                state.dataSurface->Surface(sid).Name));
                            } else {
                                ShowRecurringWarningErrorAtEnd(state,
                                    "HeatAndMoistureTransfer: Large Latent Heat Errors ", s_hbh->qvpErrReport);
                            }
                        }
                        qvp = 0.0;
                    }
                    s_hbh->thomas_d(i) += qvp;
                }
            }

            thomas_solve(s_hbh->thomas_a, s_hbh->thomas_b, s_hbh->thomas_c, s_hbh->thomas_d, s_hbh->thomas_x, N);
            for (int i = 1; i <= N; ++i) {
                s_hbh->cells(i + offset).tempp1 = s_hbh->thomas_x(i);
            }

            // Temperature bounds check — only over the solve domain. BC cells hold
            // weather/zone-air temps that EnergyPlus already validates; scanning them
            // would falsely flag legitimate winter sky temps etc. The original GS
            // path scans all cells with maxval/minval; we restrict to interior here.
            Real64 tempmax = s_hbh->cells(extIdx).tempp1;
            Real64 tempmin = tempmax;
            for (int i = extIdx + 1; i <= intIdx; ++i) {
                Real64 const t = s_hbh->cells(i).tempp1;
                if (t > tempmax) tempmax = t;
                if (t < tempmin) tempmin = t;
            }
            if (tempmax > state.dataHeatBalSurf->MaxSurfaceTempLimit) {
                if (!state.dataGlobal->WarmupFlag) {
                    if (state.dataSurface->SurfHighTempErrCount(sid) == 0) {
                        ShowSevereMessage(state,
                            EnergyPlus::format("HAMT: Temperature (high) out of bounds ({:.2R}) for surface={}",
                                               tempmax, state.dataSurface->Surface(sid).Name));
                        ShowContinueErrorTimeStamp(state, "");
                    }
                    ShowRecurringWarningErrorAtEnd(state,
                        "HAMT: Temperature Temperature (high) out of bounds; Surface=" +
                            state.dataSurface->Surface(sid).Name,
                        state.dataSurface->SurfHighTempErrCount(sid), tempmax, tempmax, _, "C", "C");
                }
            }
            if (tempmax > state.dataHeatBalSurf->MaxSurfaceTempLimitBeforeFatal) {
                if (!state.dataGlobal->WarmupFlag) {
                    ShowSevereError(state,
                        EnergyPlus::format("HAMT: HAMT: Temperature (high) out of bounds ( {:.2R}) for surface={}",
                                           tempmax, state.dataSurface->Surface(sid).Name));
                    ShowContinueErrorTimeStamp(state, "");
                    ShowFatalError(state, "Program terminates due to preceding condition.");
                }
            }
            if (tempmin < MinSurfaceTempLimit) {
                if (!state.dataGlobal->WarmupFlag) {
                    if (state.dataSurface->SurfHighTempErrCount(sid) == 0) {
                        ShowSevereMessage(state,
                            EnergyPlus::format("HAMT: Temperature (low) out of bounds ({:.2R}) for surface={}",
                                               tempmin, state.dataSurface->Surface(sid).Name));
                        ShowContinueErrorTimeStamp(state, "");
                    }
                    ShowRecurringWarningErrorAtEnd(state,
                        "HAMT: Temperature Temperature (low) out of bounds; Surface=" +
                            state.dataSurface->Surface(sid).Name,
                        state.dataSurface->SurfHighTempErrCount(sid), tempmin, tempmin, _, "C", "C");
                }
            }
            if (tempmin < MinSurfaceTempLimitBeforeFatal) {
                if (!state.dataGlobal->WarmupFlag) {
                    ShowSevereError(state,
                        EnergyPlus::format("HAMT: HAMT: Temperature (low) out of bounds ( {:.2R}) for surface={}",
                                           tempmin, state.dataSurface->Surface(sid).Name));
                    ShowContinueErrorTimeStamp(state, "");
                    ShowFatalError(state, "Program terminates due to preceding condition.");
                }
            }

            // Update vpsat and wvdc after temperature solve — solve domain only.
            // BC cell vpsat/wvdc were set once during the hoist above and stay
            // constant since BC tempp1 values don't change.
            for (int cid = extIdx; cid <= intIdx; ++cid) {
                auto &cell = s_hbh->cells(cid);
                cell.vpsat = PsyPsatFnTemp(state, cell.tempp1);
                cell.wvdc  = WVDC(cell.tempp1, baroPressOnce);
            }

            // ── Assemble moisture tridiagonal ────────────────────────────────────
            // Phase 7: hoist outer-cell-only reciprocals and the wcap/dt term.
            // Outer cell is always a solve cell (matid > 0, vtc <= 0).
            for (int i = 1; i <= N; ++i) {
                int const cid = i + offset;
                auto &cell = s_hbh->cells(cid);

                Real64 const wcap_over_dt = (cell.dwdphi > 0.0)
                                             ? (cell.dwdphi * cell.volume / s_hbh->deltat)
                                             : 0.0;
                Real64 const mu_over_wvdc = (cell.wvdc > 0.0) ? (cell.mu / cell.wvdc) : 0.0;
                Real64 const inv_dw_dwdphi = (cell.dw > 0.0 && cell.dwdphi > 0.0)
                                              ? (1.0 / (cell.dw * cell.dwdphi))
                                              : 0.0;
                bool const cellHasLiq = (inv_dw_dwdphi > 0.0);
                Real64 const cell_vpsat = cell.vpsat;

                // BDF2 time discretisation for the moisture equation — same
                // pattern as the heat tridiag above. Falls back to backward
                // Euler when no rh_prev history is available yet (first
                // step of an environment).
                Real64 b_mass_phi, d_history_phi;
                if (useBDF2 && cell.rh_prev_valid) {
                    b_mass_phi    = 1.5 * wcap_over_dt;
                    d_history_phi = wcap_over_dt * (2.0 * cell.rh - 0.5 * cell.rh_prev);
                } else {
                    b_mass_phi    = wcap_over_dt;
                    d_history_phi = wcap_over_dt * cell.rh;
                }

                s_hbh->thomas_a(i) = 0.0;
                s_hbh->thomas_b(i) = b_mass_phi;
                s_hbh->thomas_c(i) = 0.0;
                s_hbh->thomas_d(i) = d_history_phi;

                for (int ii = 1; ii <= adjmax; ++ii) {
                    int const adj = cell.adjs(ii);
                    if (adj == -1) break;
                    int const adjl = cell.adjsl(ii);
                    auto &adjCell = s_hbh->cells(adj);
                    Real64 const ovl_ii = cell.overlap(ii);
                    Real64 const inv_ovl = 1.0 / ovl_ii;
                    Real64 const dist_ii = cell.dist(ii);

                    // --- Vapor conductance ---
                    // The outer "solve" cell can be a virtual boundary cell with
                    // matid <= 0 (Extcell, Intcell), so keep the full branching;
                    // the Phase 7 hoist (mu_over_wvdc) applies in the matid>0 branch.
                    Real64 vaporr1, vaporr2;
                    if (cell.vtc > 0) {
                        vaporr1 = inv_ovl / cell.vtc;
                    } else if (cell.matid > 0) {
                        vaporr1 = dist_ii * mu_over_wvdc * inv_ovl;
                    } else {
                        vaporr1 = 0.0;
                    }
                    if (adjCell.vtc > 0) {
                        vaporr2 = inv_ovl / adjCell.vtc;
                    } else if (adjCell.matid > 0) {
                        vaporr2 = adjCell.dist(adjl) * adjCell.mu * inv_ovl / adjCell.wvdc;
                    } else {
                        vaporr2 = 0.0;
                    }

                    if (vaporr1 + vaporr2 > 0) {
                        Real64 const G_vap = 1.0 / (vaporr1 + vaporr2);
                        s_hbh->thomas_b(i) += G_vap * cell_vpsat;
                        int const local_j = adj - offset;
                        if (local_j >= 1 && local_j <= N) {
                            Real64 const psat_j = adjCell.vpsat;
                            if (local_j == i - 1) s_hbh->thomas_a(i) -= G_vap * psat_j;
                            if (local_j == i + 1) s_hbh->thomas_c(i) -= G_vap * psat_j;
                        } else {
                            // BC neighbor: move known φ_BC term to RHS (same sign as liquid BC below)
                            s_hbh->thomas_d(i) += G_vap * adjCell.vpsat * adjCell.rhp1;
                        }
                    }

                    // --- Liquid conductance ---
                    if (cellHasLiq) {
                        Real64 const rhr1 = dist_ii * inv_dw_dwdphi * inv_ovl;
                        Real64 rhr2 = 0.0;
                        if ((adjCell.dw > 0) && (adjCell.dwdphi > 0)) {
                            rhr2 = adjCell.dist(adjl) * inv_ovl / (adjCell.dw * adjCell.dwdphi);
                        }
                        // Match existing GS condition: require BOTH sides > 0 (strict product).
                        if (rhr2 > 0.0) {
                            Real64 const G_liq = 1.0 / (rhr1 + rhr2);
                            s_hbh->thomas_b(i) += G_liq;
                            int const local_j = adj - offset;
                            if (local_j >= 1 && local_j <= N) {
                                if (local_j == i - 1) s_hbh->thomas_a(i) -= G_liq;
                                if (local_j == i + 1) s_hbh->thomas_c(i) -= G_liq;
                            } else {
                                s_hbh->thomas_d(i) += G_liq * adjCell.rhp1;
                            }
                        }
                    }
                }
            }

            thomas_solve(s_hbh->thomas_a, s_hbh->thomas_b, s_hbh->thomas_c, s_hbh->thomas_d, s_hbh->thomas_x, N);

            // Apply relaxation and clamp
            for (int i = 1; i <= N; ++i) {
                Real64 const phi_new = s_hbh->thomas_x(i);
                Real64 const phi_relaxed = s_hbh->HAMTrelaxFactor * phi_new +
                                           (1.0 - s_hbh->HAMTrelaxFactor) * s_hbh->cells(i + offset).rhp1;
                s_hbh->cells(i + offset).rhp1 = std::clamp(phi_relaxed, 0.0, rhmax);
            }

            // ── Termination check ────────────────────────────────────────────
            // - Iter 1: safety check against entering state (is linearization safe?)
            // - Iter 2+: convergence check against previous iter (have we converged?)
            // The tempp1_entering / rhp1_entering buffer holds the entering state
            // on iter 1 and the previous-iter state on iter 2+. We update it
            // between iterations below.
            Real64 dT_max = 0.0, dphi_max = 0.0;
            for (int i = 1; i <= N; ++i) {
                int const cid = i + offset;
                dT_max   = std::max(dT_max,   std::abs(s_hbh->cells(cid).tempp1 - tempp1_entering(i)));
                dphi_max = std::max(dphi_max, std::abs(s_hbh->cells(cid).rhp1   - rhp1_entering(i)));
            }
            if (outer == 1) {
                if (dT_max < safetyT && dphi_max < safetyPhi) {
                    converged = true;
                    break;  // Linearization is safe; iter 1 is final answer
                }
                // Fall through to iter 2 with refreshed properties (the property
                // update at the top of the loop body uses iter-1's solved state).
            } else {
                if (dT_max < convT && dphi_max < convPhi) {
                    converged = true;
                    break;  // Picard converged at refined state
                }
            }
            // Not done — save the current iter's state as the comparison
            // basis for the next iter's convergence check.
            for (int i = 1; i <= N; ++i) {
                int const cid = i + offset;
                tempp1_entering(i) = s_hbh->cells(cid).tempp1;
                rhp1_entering(i)   = s_hbh->cells(cid).rhp1;
            }
        }  // for outer

        // ── Iteration-cap warning ─────────────────────────────────────────────
        // If the loop exhausted maxIter without satisfying a convergence criterion,
        // emit a recurring warning. This is the Thomas / BDF2 counterpart to the
        // GaussSeidel iter-cap warning above. Both reference N5 so the user knows
        // how to adjust the limit.
        if (!converged && !state.dataGlobal->WarmupFlag) {
            if (s_hbh->thomasIterCapErrCount < 16) {
                ++s_hbh->thomasIterCapErrCount;
                ShowWarningError(state,
                    std::format("HeatAndMoistureTransfer: HAMT Thomas solver reached the maximum "
                                "iteration limit ({}) for surface \"{}\"; solution may not be fully converged.",
                                maxIter, state.dataSurface->Surface(sid).Name));
                ShowContinueErrorTimeStamp(state, "");
                ShowContinueError(state,
                    "If this warning recurs, increase 'Maximum Iterations' (N5) in "
                    "HeatBalanceSettings:HeatAndMoistureTransfer, or reduce the simulation "
                    "timestep to improve per-step convergence.");
            } else {
                ShowRecurringWarningErrorAtEnd(state,
                    "HeatAndMoistureTransfer: HAMT Thomas solver reached maximum iteration limit.",
                    s_hbh->thomasIterCapErrReport);
            }
        }

        // ── Stats ─────────────────────────────────────────────────────────────
        // Cumulative counters (full-simulation diagnostics).
        s_hbh->linearizationCallCount += 1;
        s_hbh->linearizationIterTotal += nIters;
        if (nIters > s_hbh->linearizationIterMax) s_hbh->linearizationIterMax = nIters;
        // Per-surface "last iter count" — exposed via the EP output variable
        // "HAMT Surface Linearization Iterations".
        s_hbh->lastIterCount(sid) = static_cast<Real64>(nIters);
    }

    void UpdateHeatBalHAMT(EnergyPlusData &state, int const sid)
    {
        // SUBROUTINE INFORMATION:
        //       AUTHOR         Phillip Biddulph
        //       DATE WRITTEN   June 2008
        //       MODIFIED       na
        //       RE-ENGINEERED  na

        // PURPOSE OF THIS SUBROUTINE:
        // The zone heat balance equation has converged, so now the HAMT values are to be fixed
        // ready for the next iteration.
        // Fill all the report variables

        // SUBROUTINE LOCAL VARIABLE DECLARATIONS:
        Real64 watermass;
        Real64 matmass;
        // unused1208    REAL(r64), SAVE :: InOld=0.0D0
        // unused1208    REAL(r64), SAVE :: OutOld=0.0D0

        auto &s_hbh = state.dataHeatBalHAMTMgr;

        // Update Temperatures and RHs. Calculate report variables
        matmass = 0.0;
        watermass = 0.0;
        for (int cid = s_hbh->firstcell(sid); cid <= s_hbh->lastcell(sid); ++cid) {
            auto &cell = s_hbh->cells(cid);
            // fix HAMT values for this surface. The shift order matters for
            // BDF2: the OLD cell.temp (this step's T^n) becomes T^{n-1} on
            // the next step, so we copy it into cell.temp_prev *before*
            // overwriting cell.temp with the just-solved tempp1. Same for RH.
            cell.temp_prev       = cell.temp;
            cell.temp_prev_valid = true;
            cell.rh_prev         = cell.rh;
            cell.rh_prev_valid   = true;
            cell.temp = cell.tempp1;
            cell.rh = cell.rhp1;
            cell.rhp = cell.rh * 100.0;
            cell.vpreport = RHtoVP(state, cell.rh, cell.temp);
            if (cell.density > 0.0) {
                cell.wreport = cell.water / cell.density;
                watermass += (cell.water * cell.volume);
                matmass += (cell.density * cell.volume);
            }
        }

        s_hbh->watertot(sid) = 0.0;
        if (matmass > 0) {
            s_hbh->watertot(sid) = watermass / matmass;
        }

        // Phase 8: cache the converged-timestep BC values for next-step BDF2
        // extrapolation.  Shift history before storing the just-converged value:
        //   prev2 ← prev      (T^{n-1} → T^{n-2})
        //   prev  ← now       (T^n     → T^{n-1})
        // Then bump bcHistoryDepth (capped at 2). This runs after HVAC has
        // converged for this Δt, so the cache always reflects the final BC at t^n.
        s_hbh->tempOutPrev2(sid)  = s_hbh->tempOutPrev(sid);
        s_hbh->tempOutPrev(sid)   = state.dataMstBal->TempOutsideAirFD(sid);
        s_hbh->spaceMATPrev2(sid) = s_hbh->spaceMATPrev(sid);
        s_hbh->spaceMATPrev(sid)  = state.dataZoneTempPredictorCorrector
                                        ->spaceHeatBalance(state.dataSurface->Surface(sid).spaceNum).MAT;
        if (s_hbh->bcHistoryDepth(sid) < 2) s_hbh->bcHistoryDepth(sid)++;

        s_hbh->surfrh(sid) = 100.0 * s_hbh->cells(s_hbh->Intcell(sid)).rh;
        s_hbh->surfextrh(sid) = 100.0 * s_hbh->cells(s_hbh->Extcell(sid)).rh;
        s_hbh->surftemp(sid) = s_hbh->cells(s_hbh->Intcell(sid)).temp;
        s_hbh->surfexttemp(sid) = s_hbh->cells(s_hbh->Extcell(sid)).temp;
        // linearizationCallCount / linearizationIterTotal / linearizationIterMax counters are
        // maintained by CalcHeatBalHAMT_Thomas. Useful diagnostic for verifying
        // the Phase 3 linearization stays at avg ~1 iter and that the safety
        // net rarely fires. To dump them at runtime, temporarily re-add an
        // ofstream block here (see git history for the example), or expose
        // them via a proper EP output variable in a future change.
        s_hbh->surfvp(sid)    = RHtoVP(state, s_hbh->cells(s_hbh->Intcell(sid)).rh, s_hbh->cells(s_hbh->Intcell(sid)).temp);
        s_hbh->surfoutvp(sid) = RHtoVP(state, s_hbh->cells(s_hbh->Extcell(sid)).rh, s_hbh->cells(s_hbh->Extcell(sid)).temp);

        // ── Exterior net thermal radiation report (GitHub issue #11318) ──────────
        // The "Surface Outside Face Net Thermal Radiation Heat Gain Rate (per Area)"
        // output is computed in CalcOutsideSurfTemp for CTF/CondFD surfaces, but HAMT
        // bypasses that routine, so without this the report stays zero for HAMT
        // surfaces. Mirror the same formula here using the converged HAMT exterior
        // surface temperature: net LWR exchange with surrounding surfaces, air, sky,
        // and ground. SurfQRadLWOutSrdSurfs and the SurfH*Ext coefficients are
        // populated for HAMT surfaces in CalcHeatBalanceOutsideSurf.
        {
            Real64 const area  = state.dataSurface->Surface(sid).Area;
            Real64 const Tsurf = s_hbh->surfexttemp(sid);                    // HAMT exterior surface temp [C]
            Real64 const Tdb   = state.dataSurface->SurfOutDryBulbTemp(sid); // air (also approximates ground, as in CTF)
            Real64 const Tsky  = state.dataEnvrn->SkyTemp;                   // sky temp [C] (matches CalcOutsideSurfTemp)
            state.dataHeatBalSurf->SurfQdotRadOutRep(sid) =
                state.dataHeatBalSurf->SurfQRadLWOutSrdSurfs(sid) * area +
                state.dataHeatBalSurf->SurfHAirExt(sid) * area * (Tdb - Tsurf) +
                state.dataHeatBalSurf->SurfHSkyExt(sid) * area * (Tsky - Tsurf) +
                state.dataHeatBalSurf->SurfHGrdExt(sid) * area * (Tdb - Tsurf);
            state.dataHeatBalSurf->SurfQdotRadOutRepPerArea(sid) =
                (area > 0.0) ? state.dataHeatBalSurf->SurfQdotRadOutRep(sid) / area : 0.0;
        }

        // ── Heat flux at surface faces (fixes GitHub issue #3693) ─────────────────
        // Convention: qflux positive = heat flowing from exterior toward interior.
        // SurfOpaqInsFaceCondFlux: positive = heat delivered from wall into zone.
        // SurfOpaqOutFaceCondFlux: positive = heat leaving wall to exterior (= -qflux at outside face).

        // Inside face: between Intcell (virtual convection cell) and last material cell.
        {
            auto &intCell = s_hbh->cells(s_hbh->Intcell(sid));
            for (int ii = 1; ii <= adjmax; ++ii) {
                int const adj = intCell.adjs(ii);
                int const adjl = intCell.adjsl(ii);
                if (adj == -1) break;
                auto &adjCell = s_hbh->cells(adj);
                if (adjCell.matid > 0) {
                    Real64 const A = intCell.overlap(ii);
                    Real64 const thermr_int = (intCell.htc > 0.0) ? 1.0 / (A * intCell.htc) : 0.0;
                    Real64 const thermr_mat = (adjCell.wthermalc > 0.0) ? adjCell.dist(adjl) / (A * adjCell.wthermalc) : 0.0;
                    if (thermr_int + thermr_mat > 0.0) {
                        Real64 const q = (adjCell.temp - intCell.temp) / ((thermr_int + thermr_mat) * A);
                        state.dataHeatBalSurf->SurfOpaqInsFaceCondFlux(sid) = q;
                        state.dataHeatBalSurf->SurfOpaqInsFaceCond(sid) = q * state.dataSurface->Surface(sid).Area;
                        intCell.qflux = q;
                    }
                    break;
                }
            }
        }

        // Outside face: between Extcell (virtual convection cell) and first material cell.
        {
            auto &extCell = s_hbh->cells(s_hbh->Extcell(sid));
            for (int ii = 1; ii <= adjmax; ++ii) {
                int const adj = extCell.adjs(ii);
                int const adjl = extCell.adjsl(ii);
                if (adj == -1) break;
                auto &adjCell = s_hbh->cells(adj);
                if (adjCell.matid > 0) {
                    Real64 const A = extCell.overlap(ii);
                    Real64 const thermr_ext = (extCell.htc > 0.0) ? 1.0 / (A * extCell.htc) : 0.0;
                    Real64 const thermr_mat = (adjCell.wthermalc > 0.0) ? adjCell.dist(adjl) / (A * adjCell.wthermalc) : 0.0;
                    if (thermr_ext + thermr_mat > 0.0) {
                        // qflux at outside face: positive = heat flowing from exterior into wall.
                        Real64 const q = (extCell.temp - adjCell.temp) / ((thermr_ext + thermr_mat) * A);
                        extCell.qflux = q;
                        // SurfOpaqOutFaceCondFlux sign convention: positive = heat leaving wall to exterior.
                        Real64 const q_out = -q;
                        state.dataHeatBalSurf->SurfOpaqOutFaceCondFlux(sid) = q_out;
                        state.dataHeatBalSurf->SurfOpaqOutFaceCond(sid) = q_out * state.dataSurface->Surface(sid).Area;
                    }
                    break;
                }
            }
        }

        // Per-cell heat flux: Fourier's law at each material cell's interior-facing adjacency.
        // qflux positive = heat flowing from exterior toward interior at that face.
        for (int cid = s_hbh->firstcell(sid); cid <= s_hbh->lastcell(sid); ++cid) {
            auto &cell = s_hbh->cells(cid);
            if (cell.matid <= 0) continue; // Virtual boundary cells handled above.
            cell.qflux = 0.0;
            for (int ii = 1; ii <= adjmax; ++ii) {
                int const adj = cell.adjs(ii);
                int const adjl = cell.adjsl(ii);
                if (adj == -1) break;
                if (adj <= cid) continue; // Skip exterior-facing adjacency; only use interior-facing (adj > cid).
                auto &adjCell = s_hbh->cells(adj);
                Real64 const A = cell.overlap(ii);
                Real64 thermr1 = (cell.wthermalc > 0.0) ? cell.dist(ii) / (A * cell.wthermalc) : 0.0;
                Real64 thermr2;
                if (adjCell.htc > 0.0) {
                    thermr2 = 1.0 / (A * adjCell.htc);
                } else if (adjCell.matid > 0) {
                    thermr2 = adjCell.dist(adjl) / (A * adjCell.wthermalc);
                } else {
                    thermr2 = 0.0;
                }
                if (thermr1 + thermr2 > 0.0) {
                    cell.qflux = (cell.temp - adjCell.temp) / ((thermr1 + thermr2) * A);
                }
                break;
            }
        }
    }

    // Resample a single (xx, yy) table onto a uniform x-grid of FAST_NGRID points
    // spanning [xmin, xmax]. The y-values at grid points are produced by calling
    // the existing slow interp() (which does linear interpolation and linear
    // extrapolation off the table edges). After this, the FastTable can be
    // queried in O(1) by fast_interp() in the Picard inner loop.
    static void build_one_fast_table(FastTable &fast,
                                     int const n,
                                     const Array1D<Real64> &xx,
                                     const Array1D<Real64> &yy,
                                     Real64 const xmin,
                                     Real64 const xmax)
    {
        if (n < 2) {
            // No usable table; leave fast.built = false so callers can guard.
            return;
        }
        Real64 const span = xmax - xmin;
        if (span <= 0.0) {
            // Degenerate range; produce a constant table at xmin.
            fast.y.allocate(FAST_NGRID);
            Real64 y0, grad_unused;
            interp(n, xx, yy, xmin, y0, grad_unused);
            for (int i = 1; i <= FAST_NGRID; ++i) fast.y(i) = y0;
            fast.x0 = xmin;
            fast.inv_dx = 1.0; // arbitrary non-zero; fidx will be 0 by clamp
            fast.built = true;
            return;
        }
        fast.y.allocate(FAST_NGRID);
        fast.x0 = xmin;
        Real64 const dx = span / static_cast<Real64>(FAST_NGRID - 1);
        fast.inv_dx = 1.0 / dx;
        for (int i = 1; i <= FAST_NGRID; ++i) {
            Real64 const x = xmin + (i - 1) * dx;
            Real64 y, grad_unused;
            interp(n, xx, yy, x, y, grad_unused);
            fast.y(i) = y;
        }
        fast.built = true;
    }

    void BuildFastTables(MaterialHAMT &mat)
    {
        // RH-indexed tables: iso (water vs rh), mu (vapor resistance vs rh).
        // Resample over [0, rhmax] to cover the full physical RH range plus the
        // small overshoot allowed by the clamp.
        if (mat.niso >= 2) {
            build_one_fast_table(mat.isoFast, mat.niso, mat.isorh, mat.isodata, 0.0, rhmax);
        }
        if (mat.nmu >= 2) {
            build_one_fast_table(mat.muFast, mat.nmu, mat.murh, mat.mudata, 0.0, rhmax);
        }

        // Water-content-indexed tables. The water domain depends on the material
        // (range of water content the iso table covers). Use [0, max(isodata)]
        // since cell.water comes from the iso lookup; if it's slightly negative
        // we clamp at 0, and if it's slightly above we extrapolate via clamp on
        // the last segment.
        Real64 wmax = 0.0;
        for (int i = 1; i <= mat.niso; ++i) {
            if (mat.isodata(i) > wmax) wmax = mat.isodata(i);
        }
        // Bump 5% to give a small headroom for transient overshoots that the
        // slow interp would have linearly extrapolated.
        wmax *= 1.05;
        if (wmax <= 0.0) wmax = 1.0; // defensive: empty/degenerate iso table

        if (mat.nsuc >= 2) {
            build_one_fast_table(mat.sucFast, mat.nsuc, mat.sucwater, mat.sucdata, 0.0, wmax);
        }
        if (mat.nred >= 2) {
            build_one_fast_table(mat.redFast, mat.nred, mat.redwater, mat.reddata, 0.0, wmax);
        }
        if (mat.ntc >= 2) {
            build_one_fast_table(mat.tcFast, mat.ntc, mat.tcwater, mat.tcdata, 0.0, wmax);
        }
    }

    void interp(int const ndata,
                const Array1D<Real64> &xx,
                const Array1D<Real64> &yy,
                Real64 const invalue,
                Real64 &outvalue,
                ObjexxFCL::Optional<Real64> outgrad)
    {
        // SUBROUTINE INFORMATION:
        //       AUTHOR         Phillip Biddulph
        //       DATE WRITTEN   June 2008
        //       MODIFIED       na
        //       RE-ENGINEERED  na

        // PURPOSE OF THIS SUBROUTINE:
        // To find a value by searching an array and interpolating between two coordinates
        // Also returns the gradient if required.

        // METHODOLOGY EMPLOYED:
        // Simple search

        // Argument array dimensioning
        EP_SIZE_CHECK(xx, ndata);
        EP_SIZE_CHECK(yy, ndata);

        // SUBROUTINE LOCAL VARIABLE DECLARATIONS:
        Real64 xxlow;
        Real64 xxhigh;
        Real64 yylow;
        Real64 yyhigh;
        Real64 mygrad;

        mygrad = 0.0;
        outvalue = 0.0;

        if (ndata > 1) {
            xxlow = xx(1);
            yylow = yy(1);
            for (int step = 2; step <= ndata; ++step) {
                xxhigh = xx(step);
                yyhigh = yy(step);
                if (invalue <= xxhigh) {
                    break;
                }
                xxlow = xxhigh;
                yylow = yyhigh;
            }

            if (xxhigh > xxlow) {
                mygrad = (yyhigh - yylow) / (xxhigh - xxlow);
                outvalue = (invalue - xxlow) * mygrad + yylow;
                // PDB August 2009 bug fix
            } else if (std::abs(xxhigh - xxlow) < 0.0000000001) {
                outvalue = yylow;
            }
        }

        if (present(outgrad)) {
            // return gradient if required
            outgrad = mygrad;
        }
    }

    Real64 RHtoVP(EnergyPlusData &state, Real64 const RH, Real64 const Temperature)
    {
        // FUNCTION INFORMATION:
        //       AUTHOR         Phillip Biddulph
        //       DATE WRITTEN   June 2008
        //       MODIFIED       na
        //       RE-ENGINEERED  na

        // PURPOSE OF THIS FUNCTION:
        // Convert Relative Humidity and Temperature to Vapor Pressure

        // Return value
        Real64 RHtoVP;

        // FUNCTION LOCAL VARIABLE DECLARATIONS:
        Real64 VPSat;

        VPSat = PsyPsatFnTemp(state, Temperature);

        RHtoVP = RH * VPSat;

        return RHtoVP;
    }

    Real64 WVDC(Real64 const Temperature, Real64 const ambp)
    {
        // FUNCTION INFORMATION:
        //       AUTHOR         Phillip Biddulph
        //       DATE WRITTEN   June 2008
        //       MODIFIED       na
        //       RE-ENGINEERED  na

        // PURPOSE OF THIS FUNCTION:
        // To calculate the Water Vapor Diffusion Coefficient in air
        // using the temperature and ambient atmospheric pressor

        // REFERENCES:
        // K?zel, H.M. (1995) Simultaneous Heat and Moisture Transport in Building Components.
        // One- and two-dimensional calculation using simple parameters. IRB Verlag 1995

        // Return value
        Real64 WVDC;

        WVDC = (2.e-7 * std::pow(Temperature + Constant::Kelvin, 0.81)) / ambp;

        return WVDC;
    }

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

} // namespace EnergyPlus
