/*
 *          Copyright Oscar RUZ NUNES 2026.
 * Distributed under the Boost Software License, Version 1.0.
 *       (See accompanying file LICENSE or copy at
 *          https://www.boost.org/LICENSE_1_0.txt)
 */
/**
 * @file Atrium.cpp
 * @brief Driver for the 3D atrium flow and coagulation-kinetics example.
 *
 * Run:
 *   mpirun -n 8 ./examples/Heart/Atrium -atrium_mesh <path> -atrium_dt 1e-3
 *
 * Options: -atrium_mesh, -atrium_mesh_scale, -atrium_dt, -atrium_flow_cycles,
 *          -atrium_species_cycles, -atrium_pin, -atrium_steps,
 *          -atrium_order (degree of the velocity, 1 or 2),
 *          -atrium_newton (Newton passes on the convective term at each step,
 *          instead of one Oseen solve), -atrium_newton_rtol,
 *          -atrium_newton_max_it, -atrium_device (flow step assembled and
 *          solved on the Kokkos device), -atrium_pc_reuse (keep the flow
 *          preconditioner between solves, on by default),
 *          -atrium_pc_refresh_ratio (iteration growth that rebuilds it).
 */
#include <algorithm>
#include <cassert>
#include <cmath>
#include <iostream>
#include <numbers>
#include <stdexcept>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>

#include <petscsys.h>
#include <petscvec.h>

#include <Rodin/Alert.h>
#include <Rodin/Configure.h>

#ifdef RODIN_USE_SCOTCH
#include <Rodin/Scotch/MeshPartitioner.h>
#endif

#include "Atrium.h"
#ifdef RODIN_EXAMPLES_ATRIUM_DEVICE
#include "AtriumDevice.h"
#endif
#include "CoronaryArtery/CoronaryArteryAlerts.h"
#include "CoronaryArtery/CoronaryArteryTiming.h"

namespace Rodin::Examples::Heart
{
  using namespace Rodin;
  using namespace Rodin::Math;
  using namespace Rodin::Solver;
  using namespace Rodin::Geometry;
  using namespace Rodin::Variational;

  namespace
  {
    constexpr int RootRank = 0;

    void setPrefixedDefault(
      const std::string& prefix, const char* suffix, const char* value)
    {
      const std::string name = "-" + prefix + suffix;
      PetscBool set = PETSC_FALSE;
      PetscErrorCode ierr =
        PetscOptionsHasName(PETSC_NULLPTR, PETSC_NULLPTR, name.c_str(), &set);
      assert(ierr == PETSC_SUCCESS);
      if (!set)
      {
        ierr = PetscOptionsSetValue(PETSC_NULLPTR, name.c_str(), value);
        assert(ierr == PETSC_SUCCESS);
      }
      (void)ierr;
    }

    /// @brief The projections invert mass matrices: CG + Jacobi, never the
    ///        global direct solver used by the coupled flow system.
    void configureMassSolver(Rodin::Solver::KSP& ksp, const std::string& prefix)
    {
      setPrefixedDefault(prefix, "ksp_type", "cg");
      setPrefixedDefault(prefix, "pc_type", "jacobi");
      ksp.setPrefix(prefix);
    }
  }

  template <size_t K>
  Atrium<K>::Atrium(const Context::MPI& context, const Config& cfg)
    : m_cfg(cfg),
      m_input(makeInput(m_cfg)),
      m_model(m_input),
      m_mesh(makeMesh(context, m_cfg)),
      m_xdmf(context.getCommunicator(), m_cfg.xdmfBasename),
      m_inletSet(m_cfg.inlets.begin(), m_cfg.inlets.end()),
      m_vh(std::integral_constant<size_t, K>{}, m_mesh, m_mesh.getSpaceDimension()),
      m_sh(std::integral_constant<size_t, 1>{}, m_mesh),
      m_u(m_vh),
      m_p(m_sh),
      m_v(m_vh),
      m_q(m_sh),
      m_uOld(m_vh),
      m_uLin(m_vh),
      m_sTrial(m_sh),
      m_sTest(m_sh),
      m_wTrial(m_vh),
      m_wTest(m_vh),
      m_tauFn([this](const Point& p) { return vmsTauAt(p); }),
      m_tauCFn([this](const Point& p) { return tauCAt(p); }),
      m_sqrtTauCFn([this](const Point& p) { return sqrtTauCAt(p); }),
      m_tauPFn([this](const Point& p) { return tauPAt(p); }),
      m_piTilde(m_sh),
      m_convProjection(m_vh),
      m_sub(m_vh),
      m_subOld(m_vh),
      m_th(m_sh),
      m_fg(m_sh),
      m_fn(m_sh),
      m_vth(m_sh),
      m_vfg(m_sh),
      m_vfn(m_sh),
      m_thCur(m_sh),
      m_fgCur(m_sh),
      m_fnCur(m_sh),
      m_thPrev(m_sh),
      m_fgPrev(m_sh),
      m_fnPrev(m_sh),
      m_wss(m_vh),
      m_gradRec0(m_vh),
      m_gradRec1(m_vh),
      m_gradRec2(m_vh),
      m_netShear(m_vh),
      m_absShear(m_sh),
      m_shearMagnitude(m_sh),
      m_tawss(m_sh),
      m_osi(m_sh),
      m_activation(m_sh),
      m_qFlux(m_sh),
      m_one(m_sh),
      m_flux(m_qFlux),
      m_flow(m_u, m_p, m_v, m_q),
      m_flowKSP(m_flow),
      m_species(m_th, m_fg, m_fn, m_vth, m_vfg, m_vfn),
      m_speciesKSP(m_species),
      m_scalarProjection(m_sTrial, m_sTest),
      m_scalarProjectionKSP(m_scalarProjection),
      m_vectorProjection(m_wTrial, m_wTest),
      m_vectorProjectionKSP(m_vectorProjection),
      m_wssTrial(m_vh),
      m_wssTest(m_vh),
      m_wssProjection(m_wssTrial, m_wssTest),
      m_wssKSP(m_wssProjection)
  {
    // Queried on every rank, printed on one: nothing that may reduce over the
    // communicator belongs inside a root-only branch.
    const auto cellCount = m_mesh.getCellCount();
    const auto vertexCount = m_mesh.getVertexCount();
    const auto velocityDOFs = m_vh.getSize();
    const auto pressureDOFs = m_sh.getSize();

    if (isRoot())
    {
      Alert::Info() << "[mesh] cells=" << cellCount << " vertices=" << vertexCount
                    << " velocity DOFs=" << velocityDOFs
                    << " pressure DOFs=" << pressureDOFs << Alert::Raise;
    }
  }

  template <size_t K>
  Atrium<K>::~Atrium() = default;

  template <size_t K>
  bool Atrium<K>::isRoot() const
  {
    return m_mesh.getContext().getCommunicator().rank() == RootRank;
  }

  template <size_t K>
  typename Atrium<K>::MeshType Atrium<K>::makeMesh(
    const Context::MPI& context, const Config& cfg)
  {
    const auto& comm = context.getCommunicator();

    Rodin::MPI::Sharder sharder(context);
    if (comm.rank() == RootRank)
    {
      Geometry::Mesh<Context::Local> mesh;
      mesh.load(cfg.meshPath, IO::FileFormat::MEDIT);

      if (mesh.getSpaceDimension() != 3)
        throw std::runtime_error("Atrium expects a 3D tetrahedral mesh.");

      const size_t D = mesh.getDimension();
      mesh.getConnectivity().compute(D, D);
      mesh.getConnectivity().compute(D, 0);
      mesh.getConnectivity().compute(D, D - 1);
      mesh.getConnectivity().compute(D - 1, D);
      mesh.getConnectivity().compute(D - 1, 0);
      mesh.getConnectivity().compute(D - 1, 1);
      mesh.getConnectivity().compute(1, 0);

#ifdef RODIN_USE_SCOTCH
      Scotch::Partitioner partitioner(mesh);
#else
      Geometry::BalancedCompactPartitioner partitioner(mesh);
#endif
      partitioner.partition(static_cast<size_t>(comm.size()));
      sharder.shard(partitioner);
      sharder.scatter(RootRank);
    }

    MeshType mesh = sharder.gather(RootRank);
    mesh.scale(cfg.meshScale);

    const size_t D = mesh.getDimension();
    mesh.getConnectivity().compute(D, D);
    mesh.getConnectivity().compute(D, 0);
    mesh.getConnectivity().compute(D, D - 1);
    mesh.getConnectivity().compute(D - 1, D);
    mesh.getConnectivity().compute(D - 1, 0);
    mesh.getConnectivity().compute(D - 1, 1);
    mesh.getConnectivity().compute(1, 0);
    mesh.reconcile(2);
    mesh.reconcile(1);

    return mesh;
  }

  template <size_t K>
  typename Atrium<K>::Model::Input Atrium<K>::makeInput(const Config& cfg)
  {
    Model::Input input;

    input.rho = cfg.lv.rho;
    input.R0 = cfg.lv.R0;
    input.d0 = cfg.lv.d0;
    input.Es = cfg.lv.Es;
    input.mu = cfg.lv.mu;
    input.eta = cfg.lv.eta;
    input.alpha = cfg.lv.alpha;
    input.alphaR = cfg.lv.alphaR;
    input.k0 = cfg.lv.k0;
    input.sigma0 = cfg.lv.sigma0;

    input.Rp = cfg.lv.Rp;
    input.Cp = cfg.lv.Cp;
    input.Rd = cfg.lv.Rd;
    input.Cd = cfg.lv.Cd;

    input.proximalRadius = cfg.lv.proximalRadius;
    input.proximalLength = cfg.lv.proximalLength;
    input.distalRadius = cfg.lv.distalRadius;
    input.distalLength = cfg.lv.distalLength;

    input.mu_0 = cfg.lv.mu_0;
    input.mu_Inf = cfg.lv.mu_Inf;
    input.lambda = cfg.lv.lambda;
    input.n = cfg.lv.n;
    input.yasuda = cfg.lv.yasuda;

    input.Kat = cfg.lv.Kat;
    input.Kp = cfg.lv.Kp;
    input.Kar = cfg.lv.Kar;

    input.cavityCapacity = cfg.lv.cavityCapacity;
    input.localTolerance = cfg.lv.localTolerance;
    input.localMaxIterations = static_cast<size_t>(cfg.lv.localMaxIterations);
    input.localDamping = cfg.lv.localDamping;
    input.absRegularization = cfg.lv.absRegularization;

    input.pSv = [p = cfg.lv.systemicVenousPressure](Real) { return p; };
    input.pAt = [p = cfg.atrialPressure](Real t) { return atrialWave(p, t); };
    input.u = [a = cfg.activation](Real t) { return activationWave(a, t); };
    input.m0 = [low = cfg.lv.relaxationM0Low, high = cfg.lv.relaxationM0High,
                 lowEc = cfg.lv.relaxationM0LowEc,
                 highEc = cfg.lv.relaxationM0HighEc](Real ec) {
      if (highEc <= lowEc || ec >= highEc)
        return high;
      if (ec <= lowEc)
        return low;
      const Real s = (ec - lowEc) / (highEc - lowEc);
      return (1.0 - s) * low + s * high;
    };
    input.dm0 = [low = cfg.lv.relaxationM0Low, high = cfg.lv.relaxationM0High,
                  lowEc = cfg.lv.relaxationM0LowEc,
                  highEc = cfg.lv.relaxationM0HighEc](Real ec) {
      if (highEc <= lowEc || ec <= lowEc || ec >= highEc)
        return 0.0;
      return (high - low) / (highEc - lowEc);
    };

    {
      using PassiveEnergy = std::decay_t<decltype(input.passiveEnergy)>;
      typename PassiveEnergy::Parameters hp;
      hp.C0 = cfg.lv.passiveC0;
      hp.C1 = cfg.lv.passiveC1;
      hp.C2 = cfg.lv.passiveC2;
      hp.C3 = cfg.lv.passiveC3;
      input.passiveEnergy = PassiveEnergy(hp);
    }

    return input;
  }

  template <size_t K>
  Real Atrium<K>::activationWave(const Activation& cfg, Real t)
  {
    const Real T = cfg.period;
    const Real tau = t - T * std::floor(t / T);

    auto ss = [](Real s) {
      s = s < 0.0 ? 0.0 : (s > 1.0 ? 1.0 : s);
      return s * s * (3.0 - 2.0 * s);
    };

    if (tau < cfg.tRampStart)
      return 0.0;
    if (tau < cfg.tRampEnd)
    {
      return cfg.positiveValue *
        ss((tau - cfg.tRampStart) / (cfg.tRampEnd - cfg.tRampStart));
    }
    if (tau < cfg.tPlateauEnd)
      return cfg.positiveValue;
    if (tau < cfg.tRelaxEnd)
    {
      return cfg.positiveValue +
        (cfg.negativeValue - cfg.positiveValue) *
        ss((tau - cfg.tPlateauEnd) / (cfg.tRelaxEnd - cfg.tPlateauEnd));
    }
    if (tau < cfg.tNegativeEnd)
      return cfg.negativeValue;
    return cfg.negativeValue *
      (1.0 - ss((tau - cfg.tNegativeEnd) / (T - cfg.tNegativeEnd)));
  }

  template <size_t K>
  Real Atrium<K>::atrialWave(const AtrialPressure& cfg, Real t)
  {
    const Real T = cfg.period;
    const Real tau = t - T * std::floor(t / T);

    auto ss = [](Real s) {
      s = s < 0.0 ? 0.0 : (s > 1.0 ? 1.0 : s);
      return s * s * (3.0 - 2.0 * s);
    };
    auto ramp = [&](Real a, Real b, Real s) { return a + (b - a) * ss(s); };

    if (tau < cfg.t1)
      return ramp(cfg.minValue, cfg.maxValue, tau / cfg.t1);
    if (tau < cfg.t2)
      return cfg.maxValue;
    if (tau < cfg.t3)
      return ramp(cfg.maxValue, cfg.minValue, (tau - cfg.t2) / (cfg.t3 - cfg.t2));
    if (tau < cfg.t4)
      return ramp(cfg.minValue, cfg.secondThreshold, (tau - cfg.t3) / (cfg.t4 - cfg.t3));
    if (tau < cfg.t5)
      return cfg.secondThreshold;
    if (tau < cfg.t6)
      return ramp(cfg.secondThreshold, cfg.minValue, (tau - cfg.t5) / (cfg.t6 - cfg.t5));
    return cfg.minValue;
  }

  template <size_t K>
  Real Atrium<K>::cellSize(const Point& p)
  {
    return std::pow(p.getPolytope().getMeasure(), 1.0 / p.getPolytope().getDimension());
  }

  template <size_t K>
  Real Atrium<K>::viscosityAt(const Point& p) const
  {
    const auto& cy = m_cfg.viscosity;

    // Built from the same expression the momentum equation uses, so the two
    // cannot drift apart.
    const auto sym = 0.5 * (Jacobian(m_uOld) + Transpose(Jacobian(m_uOld)));
    const Real shear = std::sqrt(
      cy.gammaRegularization * cy.gammaRegularization + 2.0 * Dot(sym, sym).getValue(p));

    return cy.muInf +
      (cy.mu0 - cy.muInf) *
      std::pow(1.0 + std::pow(cy.lambda * shear, cy.yasuda), (cy.n - 1.0) / cy.yasuda);
  }

  template <size_t K>
  Real Atrium<K>::tau1At(const Point& p) const
  {
    const auto uc = m_uOld.getValue(p);
    const Real h = cellSize(p);
    const Real nu = viscosityAt(p) / m_cfg.rho;
    return 1.0 / (4.0 * nu / (h * h) + 2.0 * std::sqrt(Math::dot(uc, uc)) / h);
  }

  template <size_t K>
  Real Atrium<K>::vmsTauAt(const Point& p) const
  {
    return m_cfg.vmsScale / (m_cfg.rho / m_cfg.dt + m_cfg.rho / tau1At(p));
  }

  template <size_t K>
  Real Atrium<K>::sqrtTauCAt(const Point& p) const
  {
    const Real h = cellSize(p);
    return std::sqrt(m_cfg.gradDivScale * m_cfg.rho * h * h / (4.0 * tau1At(p)));
  }

  template <size_t K>
  Real Atrium<K>::tauCAt(const Point& p) const
  {
    const Real s = sqrtTauCAt(p);
    return s * s;
  }

  template <size_t K>
  Real Atrium<K>::tauPAt(const Point& p) const
  {
    return m_cfg.pspgScale * tau1At(p) / m_cfg.rho;
  }

  template <size_t K>
  void Atrium<K>::axpy(Real a, const ::Vec& x, ::Vec& y)
  {
    PetscErrorCode ierr = VecAXPY(y, a, x);
    assert(ierr == PETSC_SUCCESS);
    ierr = VecGhostUpdateBegin(y, INSERT_VALUES, SCATTER_FORWARD);
    assert(ierr == PETSC_SUCCESS);
    ierr = VecGhostUpdateEnd(y, INSERT_VALUES, SCATTER_FORWARD);
    assert(ierr == PETSC_SUCCESS);
    (void)ierr;
  }

  template <size_t K>
  Atrium<K>& Atrium<K>::initialize()
  {
    setupSpaces();
    setupFlow();
    if (m_cfg.device)
    {
#ifdef RODIN_EXAMPLES_ATRIUM_DEVICE
      m_device = std::make_unique<AtriumDeviceFlow<K>>(m_mesh, m_cfg, m_inletSet);
#else
      Alert::Exception() << "-atrium_device requires a build with Kokkos and PETSc."
                         << Alert::Raise;
#endif
    }
    setupSpecies();
    setupWallShear();

    if (isRoot())
    {
      m_csv.open(m_cfg.csvPath);
      if (!m_csv)
        throw std::runtime_error("Failed to open " + m_cfg.csvPath);
      writeCSVHeader();
    }

    m_initialized = true;
    return *this;
  }

  template <size_t K>
  void Atrium<K>::setupSpaces()
  {
    Model::State s0;
    s0.t = 0.0;
    s0.y = m_cfg.lv.initialY;
    s0.v = m_cfg.lv.initialV;
    s0.pv = m_input.pAt(0.0) + m_cfg.lv.initialPvOffset;
    s0.par = m_cfg.lv.initialPar;
    s0.pd = m_cfg.lv.initialPd;
    s0.w = m_input.m0(s0.ec);
    m_model.setMaxIterations(m_cfg.lv.maxIterations)
      .setAbsoluteTolerance(m_cfg.lv.absoluteTolerance)
      .setRelativeTolerance(m_cfg.lv.relativeTolerance)
      .setStepTolerance(m_cfg.lv.stepTolerance)
      .setDampingFactor(m_cfg.lv.dampingFactor);
    m_model.initialize(s0);

    const auto zeroVector = Math::SpatialVector<Real>{{0.0, 0.0, 0.0}};

    m_uOld = zeroVector;
    m_uLin = zeroVector;
    m_subOld = zeroVector;
    m_sub = zeroVector;
    m_convProjection = zeroVector;
    m_wss = zeroVector;
    m_netShear = zeroVector;
    m_gradRec0 = zeroVector;
    m_gradRec1 = zeroVector;
    m_gradRec2 = zeroVector;

    m_piTilde = Real(0);
    m_absShear = Real(0);
    m_shearMagnitude = Real(0);
    m_tawss = Real(0);
    m_osi = Real(0);
    m_activation = Real(0);
    m_one = Real(1);

    const Real fg0 =
      m_cfg.thrombosis.fibrinogenSinusRhythm / m_cfg.thrombosis.fibrinogenMolarMass;
    m_thCur = Real(0);
    m_thPrev = Real(0);
    m_fnCur = Real(0);
    m_fnPrev = Real(0);
    m_fgCur = fg0;
    m_fgPrev = fg0;

    configureMassSolver(m_scalarProjectionKSP, "atrium_sproj_");
    configureMassSolver(m_vectorProjectionKSP, "atrium_vproj_");
    configureMassSolver(m_wssKSP, "atrium_wss_");

    m_u.setName("velocity");
    m_p.setName("pressure");
    m_thCur.setName("thrombin");
    m_fgCur.setName("fibrinogen");
    m_fnCur.setName("fibrin");
    m_tawss.setName("TAWSS");
    m_osi.setName("OSI");
    m_activation.setName("activation");
    m_wss.setName("shearStress");

    m_xdmf.setMesh(m_mesh);
    m_xdmf.add("velocity", m_u.getSolution());
    m_xdmf.add("pressure", m_p.getSolution());
    m_xdmf.add("thrombin", m_thCur);
    m_xdmf.add("fibrinogen", m_fgCur);
    m_xdmf.add("fibrin", m_fnCur);
    m_xdmf.add("TAWSS", m_tawss);
    m_xdmf.add("OSI", m_osi);
    m_xdmf.add("activation", m_activation);
    m_xdmf.add("shearStress", m_wss);

    m_outletArea = boundaryArea(m_cfg.outlet);
    m_pIn = m_cfg.inletPressureMean;
    m_pOut = m_model.getState().pv;
    m_outletPressure = m_pIn;
    m_mitralZ = m_cfg.mitralResistanceOpen * m_outletArea;

    if (isRoot())
    {
      Alert::Info() << "[outlet] area = " << m_outletArea << " m^2  pv0 = " << m_pOut
                    << " Pa" << Alert::Raise;
    }
  }

  template <size_t K>
  Real Atrium<K>::boundaryArea(Attribute tag)
  {
    m_flux = BoundaryIntegral(m_one, m_qFlux).over(tag);
    m_flux.assemble();
    return std::max<Real>(m_flux(m_one), 1e-12);
  }

  template <size_t K>
  void Atrium<K>::setupFlow()
  {
    const size_t dim = m_mesh.getSpaceDimension();
    const auto normal = BoundaryNormal(m_mesh);
    const auto& cy = m_cfg.viscosity;
    const Real deltaMu = cy.mu0 - cy.muInf;
    const Real rho = m_cfg.rho;
    const Real dt = m_cfg.dt;

    const auto symU = 0.5 * (Jacobian(m_u) + Transpose(Jacobian(m_u)));
    const auto symV = 0.5 * (Jacobian(m_v) + Transpose(Jacobian(m_v)));
    const auto symLag = 0.5 * (Jacobian(m_uOld) + Transpose(Jacobian(m_uOld)));

    const auto shearLag =
      Sqrt(cy.gammaRegularization * cy.gammaRegularization + 2.0 * Dot(symLag, symLag));
    const auto muLag = cy.muInf +
      deltaMu * Pow(1.0 + Pow(cy.lambda * shearLag, cy.yasuda), (cy.n - 1.0) / cy.yasuda);

    const auto convU = Mult(Jacobian(m_u), m_uLin);
    const auto temam = Div(m_uOld) * Dot(m_u, m_v);

    const auto uNormal = Dot(m_u, normal) * normal;
    const auto uTangential = m_u - uNormal;

    const auto inletBackflow =
      0.5 * rho * m_cfg.inletBackflowStabilization * Max(Dot(m_uOld, normal), 0.0);
    const auto outletBackflow =
      0.5 * rho * m_cfg.outletBackflowStabilization * Max(-Dot(m_uOld, normal), 0.0);

    // Time-dependent scalars enter through these, so the form is assigned once
    // and only reassembled.
    RealFunction pInFn = [this](const Point&) { return m_pIn; };
    RealFunction pOutFn = [this](const Point&) { return m_pOut; };
    RealFunction mitralFn = [this](const Point&) { return m_mitralZ; };

    const auto oseen = (rho / dt) * Integral(m_u, m_v) -
      (rho / dt) * Integral(m_uOld, m_v)

      + rho * Integral(Dot(convU, m_v)) + 0.5 * rho * Integral(temam)

      // Dynamic orthogonal subscale u'^{n+1} = tau rho (Pi - (grad u^{n+1}) u^n)
      // + tau rho / dt u'^n: the residual part is implicit through the bilinear
      // integrator minus the projection, so the linear integrator only carries
      // the history tau rho / dt u'^n of the previous step. Passing m_sub
      // (already tau rho (Pi - conv) + tau rho / dt u'^n) would count the
      // orthogonal residual twice. Same form in AtriumDevice.h.
      + VMSConvectionBilinearIntegrator(m_u, m_v, m_uOld, m_tauFn, rho) -
      VMSConvectionLinearIntegrator(
        m_v, m_subOld, m_uOld, m_convProjection, m_tauFn, rho, dt)

      + VMSGradDivBilinearIntegrator(m_u, m_v, m_tauCFn) -
      VMSGradDivLinearIntegrator(m_v, m_piTilde, m_sqrtTauCFn)

      + 2.0 * Integral(muLag * symU, symV) - Integral(m_p, Div(m_v)) +
      Integral(Div(m_u), m_q) +
      m_cfg.pressurePenalty * Integral(m_p, m_q)

      // PSPG. Required by the equal-order pair.
      // The quadrature orders of this term and of the viscous one are inferred
      // as the taus and the viscosity report no degree: k_p + k_q - 2 = 0 for
      // this pair of gradients, 2K for the viscous term. AtriumDevice.h sets
      // the same orders explicitly so that both paths assemble the same system.
      + Integral(m_tauPFn * Grad(m_p), Grad(m_q))

      + BoundaryIntegral(pInFn * Dot(m_v, normal)).over(m_inletSet) +
      BoundaryIntegral(pOutFn * Dot(m_v, normal)).over(m_cfg.outlet)

      // Source impedance of the pressure inlets and mitral resistance,
      // both positive semidefinite and assembled implicitly.
      + m_cfg.inletImpedance * BoundaryIntegral(Dot(uNormal, m_v)).over(m_inletSet) +
      BoundaryIntegral(mitralFn * Dot(uNormal, m_v)).over(m_cfg.outlet)

      + m_cfg.inletTangentialDamping *
        BoundaryIntegral(Dot(uTangential, m_v)).over(m_inletSet)

      + BoundaryIntegral(inletBackflow * Dot(m_u, m_v)).over(m_inletSet) +
      BoundaryIntegral(outletBackflow * Dot(m_u, m_v)).over(m_cfg.outlet)

      + DirichletBC(m_u, Zero(dim)).on(m_cfg.wall);

    // Newton linearisation of rho (u . grad) u about the iterate u^k, of which
    // the Oseen form holds rho (u^k . grad) u: the remaining terms are
    // rho (u . grad) u^k on the left and rho (u^k . grad) u^k on the right.
    if (m_cfg.newton)
    {
      m_flow = oseen + rho * Integral(Dot(Mult(Jacobian(m_uLin), m_u), m_v)) -
        rho * Integral(Dot(Mult(Jacobian(m_uLin), m_uLin), m_v));
    }
    else
    {
      m_flow = oseen;
    }
  }

  template <size_t K>
  Real Atrium<K>::crosswind(const Point& p, Real cur, Real prev,
    const Math::SpatialVector<Real>& gradient, Real diffusivity, Real reaction) const
  {
    const Real gn = std::sqrt(Math::dot(gradient, gradient));
    if (gn < 1.0e-14)
      return 0.0;
    const auto uc = m_uOld.getValue(p);
    const Real residual = (cur - prev) / m_cfg.dt + Math::dot(uc, gradient) + reaction;
    return std::max<Real>(0.0,
      m_cfg.crosswindC * std::abs(residual) * cellSize(p) / (2.0 * gn) - diffusivity);
  }

  template <size_t K>
  void Atrium<K>::setupSpecies()
  {
    const Real Dth = m_cfg.thrombosis.diffusivityThrombin;
    const Real Dfg = m_cfg.thrombosis.diffusivityFibrinogen;
    const Real Dfn = m_cfg.thrombosis.diffusivityFibrin;
    const Real keff = m_cfg.thrombosis.reactionRate;
    const Real scale = m_cfg.thrombosis.stabilizationScale;
    const Real dt = m_cfg.dt;
    const Real fg0 =
      m_cfg.thrombosis.fibrinogenSinusRhythm / m_cfg.thrombosis.fibrinogenMolarMass;

    // Cell Peclet is of order 1e6 here: these are essentially pure advection
    // problems and their tau is set by the advective and transient scales, not
    // by the flow's tau.
    RealFunction tauThFn = [this, Dth, scale, dt](const Point& p) {
      const auto uc = m_uOld.getValue(p);
      return scale * supgTau(cellSize(p), std::sqrt(Math::dot(uc, uc)), Dth, 0.0, dt);
    };
    RealFunction tauFgFn = [this, Dfg, keff, scale, dt](const Point& p) {
      const auto uc = m_uOld.getValue(p);
      return scale *
        supgTau(cellSize(p), std::sqrt(Math::dot(uc, uc)), Dfg,
          keff * std::abs(m_thCur.getValue(p)), dt);
    };
    RealFunction tauFnFn = [this, Dfn, scale, dt](const Point& p) {
      const auto uc = m_uOld.getValue(p);
      return scale * supgTau(cellSize(p), std::sqrt(Math::dot(uc, uc)), Dfn, 0.0, dt);
    };

    // Codina crosswind: SUPG leaves the crosswind direction free, which is what
    // produces negative concentration bands across shear layers.
    RealFunction invSpeedSqFn = [this](const Point& p) {
      const auto uc = m_uOld.getValue(p);
      const Real u2 = Math::dot(uc, uc);
      return (u2 > 1.0e-20) ? 1.0 / u2 : 0.0;
    };
    RealFunction kdcThFn = [this, Dth](const Point& p) {
      return crosswind(p, m_thCur.getValue(p), m_thPrev.getValue(p),
        Grad(m_thCur).getValue(p), Dth, 0.0);
    };
    RealFunction kdcFgFn = [this, Dfg, keff](const Point& p) {
      return crosswind(p, m_fgCur.getValue(p), m_fgPrev.getValue(p),
        Grad(m_fgCur).getValue(p), Dfg, keff * m_thCur.getValue(p) * m_fgCur.getValue(p));
    };
    RealFunction kdcFnFn = [this, Dfn, keff](const Point& p) {
      return crosswind(p, m_fnCur.getValue(p), m_fnPrev.getValue(p),
        Grad(m_fnCur).getValue(p), Dfn,
        -keff * m_thCur.getValue(p) * m_fgCur.getValue(p));
    };

    const auto pTh = Dot(m_uOld, Grad(m_vth));
    const auto pFg = Dot(m_uOld, Grad(m_vfg));
    const auto pFn = Dot(m_uOld, Grad(m_vfn));

    m_species = (1.0 / dt) * Integral(m_th, m_vth) -
      (1.0 / dt) * Integral(m_thCur, m_vth) + Dth * Integral(Grad(m_th), Grad(m_vth)) +
      Integral(Dot(m_uOld, Grad(m_th)), m_vth)

      + (1.0 / dt) * Integral(m_fg, m_vfg) - (1.0 / dt) * Integral(m_fgCur, m_vfg) +
      Dfg * Integral(Grad(m_fg), Grad(m_vfg)) + Integral(Dot(m_uOld, Grad(m_fg)), m_vfg) +
      keff * Integral(m_thCur * m_fg, m_vfg)

      + (1.0 / dt) * Integral(m_fn, m_vfn) - (1.0 / dt) * Integral(m_fnCur, m_vfn) +
      Dfn * Integral(Grad(m_fn), Grad(m_vfn)) + Integral(Dot(m_uOld, Grad(m_fn)), m_vfn) -
      keff * Integral(m_thCur * m_fg, m_vfn)

      // Endothelial thrombin flux: a surface flux, armed by the cycle indices.
      - m_cfg.thrombosis.thrombinWallFlux *
        BoundaryIntegral(m_activation * m_vth).over(m_cfg.wall)

      // SUPG. Only fibrinogen has a sink proportional to its own unknown.
      + (1.0 / dt) * Integral(tauThFn * m_th, pTh) -
      (1.0 / dt) * Integral(tauThFn * m_thCur, pTh) +
      Integral(tauThFn * Dot(m_uOld, Grad(m_th)), pTh)

      + (1.0 / dt) * Integral(tauFgFn * m_fg, pFg) -
      (1.0 / dt) * Integral(tauFgFn * m_fgCur, pFg) +
      Integral(tauFgFn * Dot(m_uOld, Grad(m_fg)), pFg) +
      keff * Integral(tauFgFn * m_thCur * m_fg, pFg)

      + (1.0 / dt) * Integral(tauFnFn * m_fn, pFn) -
      (1.0 / dt) * Integral(tauFnFn * m_fnCur, pFn) +
      Integral(tauFnFn * Dot(m_uOld, Grad(m_fn)), pFn) -
      keff * Integral(tauFnFn * m_thCur * m_fg, pFn)

      // Codina crosswind, one pair per species.
      + Integral(kdcThFn * Grad(m_th), Grad(m_vth)) -
      Integral(kdcThFn * invSpeedSqFn * Dot(m_uOld, Grad(m_th)), Dot(m_uOld, Grad(m_vth)))

      + Integral(kdcFgFn * Grad(m_fg), Grad(m_vfg)) -
      Integral(kdcFgFn * invSpeedSqFn * Dot(m_uOld, Grad(m_fg)), Dot(m_uOld, Grad(m_vfg)))

      + Integral(kdcFnFn * Grad(m_fn), Grad(m_vfn)) -
      Integral(kdcFnFn * invSpeedSqFn * Dot(m_uOld, Grad(m_fn)), Dot(m_uOld, Grad(m_vfn)))

      // Pure advection needs a Dirichlet condition on the inflow boundary;
      // fresh blood carries plasma fibrinogen and no thrombin or fibrin.
      + DirichletBC(m_th, RealFunction(Real(0))).on(m_inletSet) +
      DirichletBC(m_fn, RealFunction(Real(0))).on(m_inletSet) +
      DirichletBC(m_fg, RealFunction(fg0)).on(m_inletSet);
  }

  template <size_t K>
  void Atrium<K>::setupWallShear()
  {
    const auto normal = BoundaryNormal(m_mesh);
    const auto& cy = m_cfg.viscosity;
    const auto& uSol = m_u.getSolution();

    const auto symU = 0.5 * (Jacobian(uSol) + Transpose(Jacobian(uSol)));
    const auto shear =
      Sqrt(cy.gammaRegularization * cy.gammaRegularization + 2.0 * Dot(symU, symU));
    const auto mu = cy.muInf +
      (cy.mu0 - cy.muInf) *
        Pow(1.0 + Pow(cy.lambda * shear, cy.yasuda), (cy.n - 1.0) / cy.yasuda);

    // The viscosity is factored out of the tangential projection: each
    // occurrence of mu in the expression evaluates the velocity gradient
    // on the incident cell again.
    const auto normalRows = VectorFunction(
      Dot(m_gradRec0, normal), Dot(m_gradRec1, normal), Dot(m_gradRec2, normal));
    const auto wallStress = mu * (normalRows - Dot(normalRows, normal) * normal);

    const Real reg = 1.0e-3;
    m_wssProjection = BoundaryIntegral(Dot(m_wssTrial, m_wssTest)).over(m_cfg.wall) +
      reg * Integral(Dot(m_wssTrial, m_wssTest)) -
      BoundaryIntegral(Dot(wallStress, m_wssTest)).over(m_cfg.wall);
  }

  template <size_t K>
  bool Atrium<K>::advance0D()
  {
    const auto rep = m_model.step(m_cfg.dt);
    m_pOut = m_model.getState().pv;

    if (isRoot())
    {
      const auto& s = m_model.getState();
      ZeroDInfo() << (rep.converged ? "converged" : "NOT converged")
                  << "  iter=" << rep.iterations << "  |F|=" << rep.finalResidual
                  << "  |  pv=" << s.pv << " par=" << s.par << " pd=" << s.pd << " Pa"
                  << Alert::Raise;
    }

    return rep.converged;
  }

#ifdef RODIN_EXAMPLES_ATRIUM_DEVICE
  template <size_t K>
  void Atrium<K>::deviceBegin(int pass)
  {
    if (pass == 0)
    {
      m_device->setPrevious(m_uOld.getData());
      m_device->setOpenings(m_pIn, m_pOut, m_mitralZ);
    }
    m_device->setIterate(m_uLin.getData());
    if (m_cfg.useVMS && pass == 0)
      m_device->project();
  }

  template <size_t K>
  void Atrium<K>::deviceAssemble()
  {
    m_device->assemble();
  }

  template <size_t K>
  ::KSP Atrium<K>::deviceSolve()
  {
    m_device->solve();
    m_u.getSolution().setData(m_device->getVelocity());
    m_p.getSolution().setData(m_device->getPressure());
    return m_device->getKSP();
  }

  template <size_t K>
  ::KSP Atrium<K>::deviceKSP()
  {
    return m_device->getKSP();
  }

  template <size_t K>
  void Atrium<K>::deviceEnd()
  {
    if (m_cfg.useVMS)
      m_device->advance();
  }

  template <size_t K>
  void Atrium<K>::deviceWallShear()
  {
    m_device->computeWallShear();
  }

  template <size_t K>
  void Atrium<K>::deviceFluxes()
  {
    Real pOutlet = 0.0;
    m_device->computeFluxes(m_qIn, m_qOut, pOutlet);
    m_outletPressure = pOutlet / m_outletArea;
  }

  template <size_t K>
  void Atrium<K>::devicePullWallShear()
  {
    m_wss.setData(m_device->getWallShear());
    m_shearMagnitude.setData(m_device->getShearMagnitude());
  }

  template <size_t K>
  void Atrium<K>::devicePullCycle()
  {
    m_netShear.setData(m_device->getNetShear());
    m_absShear.setData(m_device->getAbsShear());
    m_device->resetCycle();
  }
#else
  template <size_t K>
  void Atrium<K>::deviceBegin(int)
  {}

  template <size_t K>
  void Atrium<K>::deviceAssemble()
  {}

  template <size_t K>
  ::KSP Atrium<K>::deviceSolve()
  {
    return nullptr;
  }

  template <size_t K>
  ::KSP Atrium<K>::deviceKSP()
  {
    return nullptr;
  }

  template <size_t K>
  void Atrium<K>::deviceEnd()
  {}

  template <size_t K>
  void Atrium<K>::deviceWallShear()
  {}

  template <size_t K>
  void Atrium<K>::deviceFluxes()
  {}

  template <size_t K>
  void Atrium<K>::devicePullWallShear()
  {}

  template <size_t K>
  void Atrium<K>::devicePullCycle()
  {}
#endif

  template <size_t K>
  bool Atrium<K>::solveFlow()
  {
    const Real rho = m_cfg.rho;
    const Real dt = m_cfg.dt;

    // The first steps are traced projection by projection: each of these
    // assembles a mass matrix over the whole mesh, so on a fine mesh they are
    // where the time goes, and a stall must be visible where it happens.
    const bool trace = isRoot() && m_step < 3;
    const auto phase = [trace](const char* what) {
      if (trace)
      {
        ThreeDInfo() << what << " ..." << Alert::Raise;
        std::cout.flush();
      }
    };

    // One pass solves the Oseen step; under Newton the passes are repeated,
    // each from the previous iterate, until the relative update is small.
    const int passes = m_cfg.newton ? m_cfg.newtonMaxIterations : 1;
    ::KSPConvergedReason reason = KSP_CONVERGED_ITERATING;
    m_nonlinearIterations = 0;
    m_linearIterations = 0;
    m_update = 0.0;
    bool converged = !m_cfg.newton;

    for (int pass = 0; pass < passes; ++pass)
    {
      const auto vmsStart = CoronaryClock::now();

      // The projections are those of the previous step, so that they are
      // computed by the first pass only and the Newton passes solve a problem
      // whose sole nonlinearity is the convective term.
      // Only the two orthogonal-subscale projections remain: Pi[(grad u)u] and
      // Pi[sqrt(tau_C) div u]. Those ARE L2 projections -- subtracting the
      // projection of the residual is what the method does. The taus are not:
      // they are evaluated pointwise by m_tauFn and friends.
      if (m_cfg.device)
      {
        phase("VMS: projecting on the device");
        deviceBegin(pass);
      }
      // The host projections only feed the host form.
      if (m_cfg.useVMS && pass == 0 && !m_cfg.device)
      {
        phase("VMS: projecting the convective term");
        projectVector(Mult(Jacobian(m_uOld), m_uOld), m_convProjection);

        phase("VMS: projecting the dynamic subscale");
        const size_t dim = m_mesh.getSpaceDimension();
        projectVector(
          VectorFunction(dim,
            [this, rho, dt, dim](const Point& p) -> Math::SpatialVector<Real> {
              const auto conv = Mult(Jacobian(m_uOld), m_uOld).getValue(p);
              const auto proj = m_convProjection.getValue(p);
              const auto old = m_subOld.getValue(p);
              const Real tau = vmsTauAt(p);
              Math::SpatialVector<Real> out(dim);
              for (Index c = 0; c < static_cast<Index>(dim); ++c)
                out(c) = tau * rho * (old(c) / dt - (conv(c) - proj(c)));
              return out;
            }),
          m_sub);

        // Same sqrt(tau_C) that multiplies div(v) in the linear term and whose
        // square is the implicit coefficient. Lagging one of the two by a step,
        // as the reference does, leaves the explicit half larger than the
        // implicit one on the step where the viscosity drops.
        phase("VMS: projecting the grad-div residual");
        project(m_sqrtTauCFn * Div(m_uOld), m_piTilde);
      }

      m_timing.vms += secondsSince(vmsStart);

      if (isRoot())
      {
        ThreeDInfo() << "Assembling the flow system ("
                     << (m_vh.getSize() + m_sh.getSize()) << " unknowns) ..."
                     << Alert::Raise;
        std::cout.flush();
      }

      const auto assemblyStart = CoronaryClock::now();
      if (m_cfg.device)
        deviceAssemble();
      else
        m_flow.assemble();
      m_timing.assembly += secondsSince(assemblyStart);

      if (!m_cfg.device && !m_flowFieldSplitsSet)
      {
        if (isRoot())
          ThreeDInfo() << "Creating field splits ..." << Alert::Raise;
        m_flow.setFieldSplits();
        m_flowFieldSplitsSet = true;
      }

      if (isRoot())
      {
        ThreeDInfo() << "Solving with PETSc KSP"
                     << (m_step == 0 && pass == 0
                            ? " (first solve: the preconditioner is "
                              "set up here and is the slowest one)"
                            : "")
                     << " ..." << Alert::Raise;
        std::cout.flush();
      }

      const auto solveStart = CoronaryClock::now();
      ::KSP ksp = m_cfg.device ? deviceKSP() : m_flowKSP.getHandle();
      PetscErrorCode ierr = PETSC_SUCCESS;
      PetscInt iterations = 0;
      bool reused = false;
      for (;;)
      {
        // The kept preconditioner was built with the same mitral resistance:
        // the closed valve adds a Robin term several orders of magnitude
        // larger than the open one, which a kept hierarchy does not know.
        reused = m_cfg.pcReuse && m_pcBuilt && !m_pcRefresh && m_pcMitralZ == m_mitralZ;
        ierr = KSPSetReusePreconditioner(ksp, reused ? PETSC_TRUE : PETSC_FALSE);
        assert(ierr == PETSC_SUCCESS);
        // The solve sets the solver from the options again, which reaches
        // the velocity block once the splits exist.
        if (!reused && !m_cfg.velocityPCOpen.empty())
        {
          ierr = PetscOptionsSetValue(PETSC_NULLPTR, "-fieldsplit_velocity_pc_type",
            velocityPC().c_str());
          assert(ierr == PETSC_SUCCESS);
        }
        if (m_cfg.device)
          deviceSolve();
        else
          m_flow.solve(m_flowKSP);
        ierr = KSPGetConvergedReason(ksp, &reason);
        assert(ierr == PETSC_SUCCESS);
        ierr = KSPGetIterationNumber(ksp, &iterations);
        assert(ierr == PETSC_SUCCESS);
        if (!reused)
        {
          m_pcBuilt = true;
          m_pcRefresh = false;
          m_pcMitralZ = m_mitralZ;
          if (pass == 0)
            m_pcFreshIterations = iterations;
          break;
        }
        if (reason > 0)
          break;
        // A kept preconditioner that fails is rebuilt and the solve is
        // repeated from where it stopped.
        if (isRoot())
        {
          KSPInfo() << "Did NOT converge with the kept preconditioner after "
                    << iterations << " iterations, rebuilding it" << Alert::Raise;
        }
        m_linearIterations += static_cast<int>(iterations);
        m_pcRefresh = true;
      }
      // The first pass of a step is the one compared with the rebuild: the
      // Newton passes need fewer iterations as they converge.
      if (reused && pass == 0 &&
          static_cast<Real>(iterations) >
            m_cfg.pcRefreshRatio * static_cast<Real>(m_pcFreshIterations))
      {
        m_pcRefresh = true;
      }
      const Real solveTime = secondsSince(solveStart);
      m_timing.solve += solveTime;

      ierr = KSPGetResidualNorm(ksp, &m_residual);
      assert(ierr == PETSC_SUCCESS);
      m_linearIterations += static_cast<int>(iterations);
      ++m_nonlinearIterations;

      // Relative update ||u^{k+1} - u^k||_2 / ||u^{k+1}||_2 of the velocity
      // coefficients, which is the stopping measure of the Newton passes.
      ::Vec& solution = m_u.getSolution().getData();
      ::Vec difference;
      ierr = VecDuplicate(solution, &difference);
      assert(ierr == PETSC_SUCCESS);
      ierr = VecCopy(solution, difference);
      assert(ierr == PETSC_SUCCESS);
      ierr = VecAXPY(difference, -1.0, m_uLin.getData());
      assert(ierr == PETSC_SUCCESS);
      PetscReal differenceNorm = 0.0;
      PetscReal solutionNorm = 0.0;
      ierr = VecNorm(difference, NORM_2, &differenceNorm);
      assert(ierr == PETSC_SUCCESS);
      ierr = VecNorm(solution, NORM_2, &solutionNorm);
      assert(ierr == PETSC_SUCCESS);
      ierr = VecDestroy(&difference);
      assert(ierr == PETSC_SUCCESS);
      (void)ierr;
      m_update = solutionNorm > 0.0 ? differenceNorm / solutionNorm : 0.0;

      if (isRoot())
      {
        KSPInfo() << (reason > 0 ? "Converged" : "Did NOT converge")
                  << "  iterations = " << iterations << "  (" << solveTime << " s)"
                  << "  pc " << (reused ? "kept" : "built")
                  << (m_cfg.velocityPCOpen.empty() ? "" : " velocity " + velocityPC())
                  << "  pass " << (pass + 1) << "  update = " << m_update << Alert::Raise;
      }

      m_uLin.setData(solution);

      if (reason <= 0 || !std::isfinite(m_update))
        break;
      if (m_cfg.newton && m_update <= m_cfg.newtonRelativeTolerance)
      {
        converged = true;
        break;
      }
    }

    m_uOld.setData(m_u.getSolution().getData());
    if (m_cfg.device)
      deviceEnd();
    else if (m_cfg.useVMS)
      m_subOld.setData(m_sub.getData());

    // A direct solve always "succeeds": check the physics, not the solver.
    m_speed = std::max(std::abs(m_uOld.max()), std::abs(m_uOld.min()));

    if (!converged && isRoot())
    {
      Alert::Warning() << "[flow] Newton did not reach the relative update "
                       << m_cfg.newtonRelativeTolerance << " in " << m_nonlinearIterations
                       << " passes (update = " << m_update << ")" << Alert::Raise;
    }

    return converged && reason > 0 && std::isfinite(m_speed) &&
      m_speed <= m_cfg.maxVelocity;
  }

  template <size_t K>
  void Atrium<K>::computeWallShear()
  {
    const auto shearStart = CoronaryClock::now();
    const auto& uSol = m_u.getSolution();

    if (isRoot() && m_step < 3)
    {
      ThreeDInfo() << "Recovering the wall shear stress ..." << Alert::Raise;
      std::cout.flush();
    }

    if (m_cfg.device)
    {
      // Recovery, magnitude and accumulators all live on the device; the host
      // fields are refreshed when they are written or when the cycle closes.
      deviceWallShear();
      PetscErrorCode ierr = VecMax(m_device->getShearMagnitude(), PETSC_NULLPTR, &m_maxShear);
      assert(ierr == PETSC_SUCCESS);
      (void)ierr;
      m_timing.shear = secondsSince(shearStart);
      return;
    }

    projectVector(VectorFunction(Component(Jacobian(uSol), 0, 0),
                    Component(Jacobian(uSol), 0, 1), Component(Jacobian(uSol), 0, 2)),
      m_gradRec0);
    projectVector(VectorFunction(Component(Jacobian(uSol), 1, 0),
                    Component(Jacobian(uSol), 1, 1), Component(Jacobian(uSol), 1, 2)),
      m_gradRec1);
    projectVector(VectorFunction(Component(Jacobian(uSol), 2, 0),
                    Component(Jacobian(uSol), 2, 1), Component(Jacobian(uSol), 2, 2)),
      m_gradRec2);

    m_wssProjection.assemble();
    m_wssProjection.solve(m_wssKSP);
    m_wss.setData(m_wssTrial.getSolution().getData());

    // Cycle accumulators. Two are needed and they are not interchangeable: the
    // vector integral measures how much net direction survives, the scalar one
    // how much shear was applied regardless of direction.
    axpy(m_cfg.dt, m_wss.getData(), m_netShear.getData());

    // Nodal interpolation, NOT an L2 projection. |tau_w| is finite on the wall
    // and zero in the interior; the L2 projection of that onto a continuous P1
    // field undershoots, and a single negative node makes int|tau_w| dt -- and
    // therefore TAWSS, which is non-negative by definition -- come out
    // negative. Interpolating the magnitude at the nodes cannot.
    m_shearMagnitude.project(Sqrt(Dot(m_wss, m_wss)));
    axpy(m_cfg.dt, m_shearMagnitude.getData(), m_absShear.getData());
    m_maxShear = m_shearMagnitude.max();

    m_timing.shear = secondsSince(shearStart);
  }

  template <size_t K>
  void Atrium<K>::closeCycle(Real elapsed)
  {
    if (elapsed <= 0.0)
      return;

    // TAWSS = (1/T) int |tau_w| dt. An exact scaling of the accumulator: no
    // projection, no interpolation, no way to change its sign.
    PetscErrorCode ierr = VecCopy(m_absShear.getData(), m_tawss.getData());
    assert(ierr == PETSC_SUCCESS);
    ierr = VecScale(m_tawss.getData(), 1.0 / elapsed);
    assert(ierr == PETSC_SUCCESS);
    ierr = VecGhostUpdateBegin(m_tawss.getData(), INSERT_VALUES, SCATTER_FORWARD);
    assert(ierr == PETSC_SUCCESS);
    ierr = VecGhostUpdateEnd(m_tawss.getData(), INSERT_VALUES, SCATTER_FORWARD);
    assert(ierr == PETSC_SUCCESS);
    (void)ierr;

    // OSI = (1/2)[1 - |int tau_w dt| / int |tau_w| dt]. Both accumulators are
    // built from the same nodal values, so the triangle inequality holds node
    // by node and OSI lands in [0, 1/2] on its own; the clamp is insurance.
    m_osi.project(RealFunction([this](const Point& p) -> Real {
      const Real abs = m_absShear.getValue(p);
      if (abs <= 0.0)
        return 0.0;
      const auto net = m_netShear.getValue(p);
      const Real mag = std::sqrt(Math::dot(net, net));
      return std::clamp<Real>(0.5 * (1.0 - mag / abs), 0.0, 0.5);
    }));

    // Smooth, bounded activation: a logistic in the measured shear threshold
    // times the oscillatory index mapped onto [0,1]. Clamped to [0,1] so the
    // endothelial thrombin flux can never turn into a sink -- a negative
    // activation is what drives thrombin, and with it fibrin, negative.
    const Real tauA = m_cfg.thrombosis.activationShearStress;
    const Real width = std::max<Real>(m_cfg.thrombosis.activationShearWidth, 1e-12);
    m_activation.project(RealFunction([this, tauA, width](const Point& p) -> Real {
      const Real low = 1.0 / (1.0 + std::exp((m_tawss.getValue(p) - tauA) / width));
      const Real osi = std::clamp<Real>(2.0 * m_osi.getValue(p), 0.0, 1.0);
      return std::clamp<Real>(low * osi, 0.0, 1.0);
    }));

    // The ghost entries must be zeroed too, or the next accumulation reads a
    // stale halo.
    const auto zeroWithGhosts = [](::Vec& v) {
      PetscErrorCode e = VecZeroEntries(v);
      assert(e == PETSC_SUCCESS);
      e = VecGhostUpdateBegin(v, INSERT_VALUES, SCATTER_FORWARD);
      assert(e == PETSC_SUCCESS);
      e = VecGhostUpdateEnd(v, INSERT_VALUES, SCATTER_FORWARD);
      assert(e == PETSC_SUCCESS);
      (void)e;
    };

    zeroWithGhosts(m_netShear.getData());
    zeroWithGhosts(m_absShear.getData());
  }

  template <size_t K>
  void Atrium<K>::solveSpecies()
  {
    const auto speciesStart = CoronaryClock::now();

    if (isRoot() && m_step < 3)
    {
      ThreeDInfo() << "Assembling and solving the coagulation kinetics ..."
                   << Alert::Raise;
      std::cout.flush();
    }

    m_species.assemble();
    m_species.solve(m_speciesKSP);

    // History rotates before the current level is overwritten: the crosswind
    // residual needs both c^n and c^{n-1}.
    m_thPrev.setData(m_thCur.getData());
    m_fgPrev.setData(m_fgCur.getData());
    m_fnPrev.setData(m_fnCur.getData());
    m_thCur.setData(m_th.getSolution().getData());
    m_fgCur.setData(m_fg.getSolution().getData());
    m_fnCur.setData(m_fn.getSolution().getData());

    m_timing.species = secondsSince(speciesStart);
  }

  template <size_t K>
  void Atrium<K>::computeFluxes()
  {
    if (m_cfg.device)
    {
      deviceFluxes();
      return;
    }

    const auto normal = BoundaryNormal(m_mesh);
    const auto& uSol = m_u.getSolution();

    m_flux = BoundaryIntegral(Dot(uSol, normal), m_qFlux).over(m_inletSet);
    m_flux.assemble();
    m_qIn = m_flux(m_one);

    m_flux = BoundaryIntegral(Dot(uSol, normal), m_qFlux).over(m_cfg.outlet);
    m_flux.assemble();
    m_qOut = m_flux(m_one);

    m_flux = BoundaryIntegral(m_p.getSolution(), m_qFlux).over(m_cfg.outlet);
    m_flux.assemble();
    m_outletPressure = m_flux(m_one) / m_outletArea;
  }

  template <size_t K>
  void Atrium<K>::writeCSVHeader()
  {
    m_csv << "t,cycle,pIn,pv,par,pd,valveOpen,qIn,qOut,maxU,"
          << "maxTAWSS,maxOSI,maxActivation,maxThrombin,minFibrinogen,maxFibrin\n";
  }

  template <size_t K>
  void Atrium<K>::writeCSVRow(int cycle)
  {
    const auto& s = m_model.getState();
    const Real tawss = m_tawss.max();
    const Real osi = m_osi.max();
    const Real activation = m_activation.max();
    const Real th = m_thCur.max();
    const Real fg = m_fgCur.min();
    const Real fn = m_fnCur.max();

    if (!isRoot())
      return;

    m_csv << m_t << ',' << cycle << ',' << m_pIn << ',' << s.pv << ',' << s.par << ','
          << s.pd << ',' << (m_valveOpen ? 1 : 0) << ',' << m_qIn << ',' << m_qOut << ','
          << m_speed << ',' << tawss << ',' << osi << ',' << activation << ',' << th
          << ',' << fg << ',' << fn << '\n';
    m_csv.flush();
  }

  template <size_t K>
  int Atrium<K>::run()
  {
    if (!m_initialized)
      initialize();

    const Real PI = std::numbers::pi_v<Real>;
    const int stepsPerCycle = static_cast<int>(m_cfg.period / m_cfg.dt + 0.5);
    const int totalCycles = m_cfg.flowCycles + m_cfg.speciesCycles;
    const int totalSteps = m_cfg.maxSteps
      ? std::min(*m_cfg.maxSteps, totalCycles * stepsPerCycle)
      : totalCycles * stepsPerCycle;

    if (isRoot())
    {
      Alert::Info() << "[run] " << totalCycles << " cycles of " << stepsPerCycle
                    << " steps (" << totalSteps << " total); species from cycle "
                    << (m_cfg.flowCycles + 1) << " on; XDMF every " << m_cfg.outputEvery
                    << " steps and at every cycle boundary" << Alert::Raise;
    }

    Real cycleElapsed = 0.0;
    const auto runStart = CoronaryClock::now();

    for (int step = 0; step < totalSteps; ++step)
    {
      m_step = step;
      m_timing = Timing{};
      const auto stepStart = CoronaryClock::now();

      m_t += m_cfg.dt;
      const int cycle = step / stepsPerCycle;
      const bool endOfCycle = (step % stepsPerCycle == stepsPerCycle - 1);

      if (isRoot())
      {
        Alert::Info() << "---- Step " << (step + 1) << "/" << totalSteps << "  cycle "
                      << (cycle + 1) << "/" << totalCycles << "  t = " << m_t
                      << " s  (dt = " << m_cfg.dt << " s"
                      << (cycle < m_cfg.flowCycles ? ", warm-up" : "") << ") ----"
                      << Alert::Raise;
        std::cout.flush();
      }

      const auto zeroDStart = CoronaryClock::now();
      const bool advanced = advance0D();
      m_timing.zeroD = secondsSince(zeroDStart);

      if (!advanced)
      {
        Alert::Exception() << "[0D] Newton did not converge at t = " << m_t
                           << Alert::Raise;
        return 1;
      }

      m_pIn = m_cfg.inletPressureMean +
        m_cfg.inletPressureAmplitude * std::sin(2.0 * PI * m_t / m_cfg.period);

      // Mitral diode, switched on the previous step's outlet pressure.
      m_valveOpen = (m_outletPressure > m_pOut);
      m_mitralZ =
        (m_valveOpen ? m_cfg.mitralResistanceOpen : m_cfg.mitralResistanceClosed) *
        m_outletArea;

      if (!solveFlow())
      {
        Alert::Exception() << "[flow] diverged at step " << (step + 1)
                           << ": max|u| = " << m_speed << " m/s" << Alert::Raise;
        return 1;
      }

      computeWallShear();
      cycleElapsed += m_cfg.dt;

      const auto fluxStart = CoronaryClock::now();
      computeFluxes();
      m_timing.fluxes = secondsSince(fluxStart);

      if (endOfCycle)
      {
        if (m_cfg.device)
          devicePullCycle();
        closeCycle(cycleElapsed);
        cycleElapsed = 0.0;

        // VecMax is collective: every rank must reach it, so the reductions
        // happen outside the root-only print.
        const Real maxTawss = m_tawss.max();
        const Real maxOsi = m_osi.max();
        const Real maxActivation = m_activation.max();

        if (isRoot())
        {
          Alert::Info() << "[cycle " << (cycle + 1) << "/" << totalCycles
                        << "] maxTAWSS=" << maxTawss << " Pa  maxOSI=" << maxOsi
                        << "  maxActivation=" << maxActivation << Alert::Raise;
        }
      }

      if (m_cfg.solveKinetics && cycle >= m_cfg.flowCycles)
        solveSpecies();

      writeCSVRow(cycle);

      // The flow is written from the very first step, so the warm-up cycles are
      // available before the species are switched on.
      const bool output =
        endOfCycle || (m_cfg.outputEvery > 0 && step % m_cfg.outputEvery == 0);
      if (output)
      {
        const auto outputStart = CoronaryClock::now();
        if (m_cfg.device)
          devicePullWallShear();
        m_xdmf.write(m_t).flush();
        m_timing.output = secondsSince(outputStart);
      }

      m_timing.total = secondsSince(stepStart);

      if (isRoot())
      {
        const Real elapsed = secondsSince(runStart);
        const Real perStep = elapsed / static_cast<Real>(step + 1);

        Alert::Info() << "[3D] max|u|=" << m_speed << " m/s  pIn=" << m_pIn
                      << " Pa  pv=" << m_pOut
                      << " Pa  valve=" << (m_valveOpen ? "open" : "closed")
                      << "  qIn=" << m_qIn << "  qOut=" << m_qOut << " m^3/s"
                      << "  max|tau_w|=" << m_maxShear << " Pa" << Alert::Raise;

        Alert::Info() << "[solver] passes=" << m_nonlinearIterations
                      << "  linear iterations=" << m_linearIterations
                      << "  residual=" << m_residual << "  update=" << m_update
                      << Alert::Raise;

        Alert::Info() << "[timing] 0D=" << m_timing.zeroD << "  vms=" << m_timing.vms
                      << "  asm=" << m_timing.assembly << "  ksp=" << m_timing.solve
                      << "  wss=" << m_timing.shear << "  flux=" << m_timing.fluxes
                      << "  species=" << m_timing.species << "  out=" << m_timing.output
                      << "  total=" << m_timing.total << " s  |  ETA "
                      << (perStep * (totalSteps - step - 1) / 60.0) << " min"
                      << Alert::Raise;
        std::cout.flush();
      }
    }

    m_xdmf.close();
    if (isRoot())
      m_csv.close();

    return 0;
  }

  template class Atrium<1>;
  template class Atrium<2>;
}

int main(int argc, char** argv)
{
#ifdef RODIN_EXAMPLES_ATRIUM_DEVICE
  // Kokkos is started before PETSc and outlives it.
  const ::Kokkos::ScopeGuard kokkos(argc, argv);
#endif
  PetscInitialize(&argc, &argv, PETSC_NULLPTR, PETSC_NULLPTR);

  const auto setPETScDefault = [](const char* name, const char* value) {
    PetscBool set = PETSC_FALSE;
    PetscErrorCode ierr = PetscOptionsHasName(PETSC_NULLPTR, PETSC_NULLPTR, name, &set);
    if (ierr == PETSC_SUCCESS && !set)
      ierr = PetscOptionsSetValue(PETSC_NULLPTR, name, value);
    assert(ierr == PETSC_SUCCESS);
    (void)ierr;
  };

  // Default flow solver: flexible GMRES on the full Schur factorization of
  // the velocity-pressure splits, the Schur complement approximated by
  // A10 diag(A00)^-1 A01 and preconditioned by one GAMG V-cycle. On the P1P1
  // atrium at 1800 Pa it gives the same max|u|, fluxes and wall shear as the
  // MUMPS LU to six digits. The velocity block takes one GAMG V-cycle while
  // the mitral valve is open and block Jacobi with ILU(0) on each process
  // while it is closed: the closed valve adds a large n n^T Robin block to
  // the outlet velocity rows, which the Chebyshev/Jacobi smoothers of GAMG do
  // not see on the component-major numbering (fgmres stalls at the closure,
  // step 175, l1 Jacobi included), and ILU is several times slower than GAMG
  // on a GPU. Naming -fieldsplit_velocity_pc_type keeps one preconditioner
  // for the whole run. The command line still wins: the direct solver is
  // back with -pc_type lu -pc_factor_mat_solver_type mumps, in which case
  // the field split options below are left unused.
  const bool directSolver = [] {
    PetscBool lu = PETSC_FALSE;
    [[maybe_unused]] PetscErrorCode ierr =
      PetscOptionsHasName(PETSC_NULLPTR, PETSC_NULLPTR, "-pc_factor_mat_solver_type", &lu);
    assert(ierr == PETSC_SUCCESS);
    return lu == PETSC_TRUE;
  }();
  std::string velocityPCOpen;
  std::string velocityPCClosed;
  if (directSolver)
  {
    setPETScDefault("-ksp_type", "preonly");
    setPETScDefault("-pc_type", "lu");
    setPETScDefault("-mat_mumps_icntl_20", "0");
    setPETScDefault("-mat_mumps_icntl_21", "0");
  }
  else
  {
    setPETScDefault("-ksp_type", "fgmres");
    setPETScDefault("-ksp_rtol", "1e-10");
    setPETScDefault("-pc_type", "fieldsplit");
    setPETScDefault("-pc_fieldsplit_type", "schur");
    setPETScDefault("-pc_fieldsplit_schur_fact_type", "full");
    setPETScDefault("-pc_fieldsplit_schur_precondition", "selfp");
    // The splits carry the names of the trial functions on both paths.
    setPETScDefault("-fieldsplit_velocity_ksp_type", "preonly");
    PetscBool velocitySet = PETSC_FALSE;
    [[maybe_unused]] PetscErrorCode ierr = PetscOptionsHasName(
      PETSC_NULLPTR, PETSC_NULLPTR, "-fieldsplit_velocity_pc_type", &velocitySet);
    assert(ierr == PETSC_SUCCESS);
    if (velocitySet == PETSC_FALSE)
    {
      velocityPCOpen = "gamg";
      velocityPCClosed = "bjacobi";
    }
    setPETScDefault("-fieldsplit_velocity_pc_type", "gamg");
    setPETScDefault("-fieldsplit_velocity_sub_pc_type", "ilu");
#ifdef RODIN_EXAMPLES_ATRIUM_DEVICE
    // The device matrices are factored and solved by Kokkos Kernels.
    PetscBool device = PETSC_FALSE;
    PetscOptionsGetBool(PETSC_NULLPTR, PETSC_NULLPTR, "-atrium_device", &device, PETSC_NULLPTR);
    if (device == PETSC_TRUE)
      setPETScDefault("-fieldsplit_velocity_sub_pc_factor_mat_solver_type", "kokkos");
#endif
    setPETScDefault("-fieldsplit_pressure_ksp_type", "preonly");
    setPETScDefault("-fieldsplit_pressure_pc_type", "gamg");
    // The aggressive coarsening squares the graph of the selfp Schur
    // complement, which is denser than the velocity block: on a MI210 the
    // product took 3.3 of the 4.8 s of each preconditioner rebuild, and the
    // plain aggregation also converges in fewer iterations.
    setPETScDefault("-fieldsplit_pressure_pc_gamg_aggressive_coarsening", "0");
  }

  boost::mpi::environment env(argc, argv);
  boost::mpi::communicator world(PETSC_COMM_WORLD, boost::mpi::comm_attach);
  Rodin::Context::MPI context(env, world);

  try
  {
    int status = 0;

    {
      Rodin::Examples::Heart::AtriumBase::Config cfg;

      char buffer[512];
      PetscBool got = PETSC_FALSE;
      PetscOptionsGetString(
        PETSC_NULLPTR, PETSC_NULLPTR, "-atrium_mesh", buffer, sizeof(buffer), &got);
      if (got)
        cfg.meshPath = buffer;

      PetscReal real = 0.0;
      got = PETSC_FALSE;
      PetscOptionsGetReal(
        PETSC_NULLPTR, PETSC_NULLPTR, "-atrium_mesh_scale", &real, &got);
      if (got)
        cfg.meshScale = real;

      got = PETSC_FALSE;
      PetscOptionsGetReal(PETSC_NULLPTR, PETSC_NULLPTR, "-atrium_dt", &real, &got);
      if (got)
        cfg.dt = real;

      got = PETSC_FALSE;
      PetscOptionsGetReal(PETSC_NULLPTR, PETSC_NULLPTR, "-atrium_pin", &real, &got);
      if (got)
        cfg.inletPressureMean = real;

      PetscInt integer = 0;
      got = PETSC_FALSE;
      PetscOptionsGetInt(
        PETSC_NULLPTR, PETSC_NULLPTR, "-atrium_flow_cycles", &integer, &got);
      if (got)
        cfg.flowCycles = static_cast<int>(integer);

      got = PETSC_FALSE;
      PetscOptionsGetInt(
        PETSC_NULLPTR, PETSC_NULLPTR, "-atrium_species_cycles", &integer, &got);
      if (got)
        cfg.speciesCycles = static_cast<int>(integer);

      got = PETSC_FALSE;
      PetscOptionsGetInt(
        PETSC_NULLPTR, PETSC_NULLPTR, "-atrium_output_every", &integer, &got);
      if (got)
        cfg.outputEvery = static_cast<int>(integer);

      got = PETSC_FALSE;
      PetscOptionsGetReal(PETSC_NULLPTR, PETSC_NULLPTR, "-atrium_vms_scale", &real, &got);
      if (got)
        cfg.vmsScale = real;

      got = PETSC_FALSE;
      PetscOptionsGetReal(
        PETSC_NULLPTR, PETSC_NULLPTR, "-atrium_graddiv_scale", &real, &got);
      if (got)
        cfg.gradDivScale = real;

      got = PETSC_FALSE;
      PetscOptionsGetReal(
        PETSC_NULLPTR, PETSC_NULLPTR, "-atrium_pspg_scale", &real, &got);
      if (got)
        cfg.pspgScale = real;

      PetscBool flag = PETSC_FALSE;
      got = PETSC_FALSE;
      PetscOptionsGetBool(PETSC_NULLPTR, PETSC_NULLPTR, "-atrium_vms", &flag, &got);
      if (got)
        cfg.useVMS = (flag == PETSC_TRUE);

      got = PETSC_FALSE;
      PetscOptionsGetBool(PETSC_NULLPTR, PETSC_NULLPTR, "-atrium_kinetics", &flag, &got);
      if (got)
        cfg.solveKinetics = (flag == PETSC_TRUE);

      got = PETSC_FALSE;
      PetscOptionsGetBool(PETSC_NULLPTR, PETSC_NULLPTR, "-atrium_newton", &flag, &got);
      if (got)
        cfg.newton = (flag == PETSC_TRUE);

      got = PETSC_FALSE;
      PetscOptionsGetReal(
        PETSC_NULLPTR, PETSC_NULLPTR, "-atrium_newton_rtol", &real, &got);
      if (got)
        cfg.newtonRelativeTolerance = real;

      got = PETSC_FALSE;
      PetscOptionsGetInt(
        PETSC_NULLPTR, PETSC_NULLPTR, "-atrium_newton_max_it", &integer, &got);
      if (got)
        cfg.newtonMaxIterations = static_cast<int>(integer);

      got = PETSC_FALSE;
      PetscOptionsGetBool(PETSC_NULLPTR, PETSC_NULLPTR, "-atrium_pc_reuse", &flag, &got);
      if (got)
        cfg.pcReuse = (flag == PETSC_TRUE);

      got = PETSC_FALSE;
      PetscOptionsGetReal(
        PETSC_NULLPTR, PETSC_NULLPTR, "-atrium_pc_refresh_ratio", &real, &got);
      if (got)
        cfg.pcRefreshRatio = real;

      cfg.velocityPCOpen = velocityPCOpen;
      cfg.velocityPCClosed = velocityPCClosed;

      got = PETSC_FALSE;
      PetscOptionsGetInt(PETSC_NULLPTR, PETSC_NULLPTR, "-atrium_steps", &integer, &got);
      if (got)
        cfg.maxSteps = static_cast<int>(integer);

      PetscOptionsGetBool(PETSC_NULLPTR, PETSC_NULLPTR, "-atrium_device", &flag, &got);
      if (got)
        cfg.device = (flag == PETSC_TRUE);

      // Degree of the velocity: 1 is the stabilised P1-P1 pair, 2 is P2-P1.
      PetscInt order = 1;
      PetscOptionsGetInt(PETSC_NULLPTR, PETSC_NULLPTR, "-atrium_order", &order, &got);
      if (order == 1)
      {
        Rodin::Examples::Heart::Atrium<1> simulation(context, cfg);
        status = simulation.initialize().run();
      }
      else if (order == 2)
      {
        Rodin::Examples::Heart::Atrium<2> simulation(context, cfg);
        status = simulation.initialize().run();
      }
      else
      {
        Rodin::Alert::Exception()
          << "-atrium_order must be 1 or 2, got " << order << Rodin::Alert::Raise;
      }
    }

    PetscFinalize();
    return status;
  }
  catch (const std::exception& e)
  {
    std::cerr << "Fatal error: " << e.what() << "\n";
    PetscFinalize();
    return 1;
  }
}
