/*
 *          Copyright Carlos BRITO PACHECO 2021 - 2026.
 * Distributed under the Boost Software License, Version 1.0.
 *       (See accompanying file LICENSE or copy at
 *          https://www.boost.org/LICENSE_1_0.txt)
 */
#ifndef EXAMPLES_HEART_ATRIUMDEVICE_H
#define EXAMPLES_HEART_ATRIUMDEVICE_H

#include <type_traits>
#include <utility>

#include <Rodin/Kokkos/MPI.h>
#include <Rodin/PETSc/Kokkos.h>

#include "Atrium.h"

namespace Rodin::Examples::Heart
{
#if defined(KOKKOS_ENABLE_CUDA)
  /// @brief Execution backend of the device flow.
  using AtriumBackend = Rodin::Context::CUDA;
#elif defined(KOKKOS_ENABLE_HIP)
  /// @brief Execution backend of the device flow.
  using AtriumBackend = Rodin::Context::HIP;
#elif defined(KOKKOS_ENABLE_SYCL)
  /// @brief Execution backend of the device flow.
  using AtriumBackend = Rodin::Context::SYCL;
#else
  /// @brief Execution backend of the device flow.
  using AtriumBackend = ::Kokkos::DefaultHostExecutionSpace;
#endif

  /// @brief Size of a tetrahedron, the cube root of its volume.
  struct AtriumCellSize
  {
    /// @param p Point of a tetrahedron. @return Size of the tetrahedron.
    KOKKOS_INLINE_FUNCTION Rodin::Real operator()(
      const Rodin::Kokkos::View<Rodin::Geometry::Point>& p) const
    {
      KOKKOS_ASSERT(p.getPolytope().getGeometry() ==
        Rodin::Geometry::Polytope::Type::Tetrahedron);
      return ::Kokkos::cbrt(p.getDistortion() / 6);
    }
  };

  /**
   * @brief Flow step of the atrium assembled and solved on a Kokkos device.
   *
   * This is the coupled velocity and pressure problem of Atrium, with the
   * same terms and the same coefficients, written with expressions the
   * device evaluates: the stabilisation parameters, which the host computes
   * in callbacks, are functions of the previous velocity here, and the
   * orthogonal-subscale terms are integrals of such functions. The quadrature
   * of those terms is the one the device chooses for each integrand.
   *
   * The device spaces number their degrees of freedom as the host spaces do,
   * so the velocity and the pressure are exchanged with the host as vectors.
   *
   * @tparam K Degree of the velocity space.
   */
  template <size_t K>
  class AtriumDeviceFlow
  {
    public:
      /// @brief Real number type.
      using Real = Rodin::Real;

      /// @brief Configuration of the atrium.
      using Config = AtriumBase::Config;

      /// @brief Distributed host mesh.
      using HostMeshType = AtriumBase::MeshType;

      /// @brief Distributed device context.
      using ContextType = Rodin::Context::Kokkos<Rodin::Context::MPI, AtriumBackend>;

      /// @brief Distributed device mesh.
      using MeshType = Rodin::Geometry::Mesh<ContextType>;

      /// @brief Velocity space.
      using VectorFESType =
        Rodin::Variational::H1<K, Rodin::Math::SpatialVector<Real>, MeshType>;

      /// @brief Pressure space.
      using ScalarFESType = Rodin::Variational::H1<1, Real, MeshType>;

      /// @brief Velocity field.
      using VectorGridFunctionType =
        decltype(Rodin::PETSc::Variational::GridFunction(
          std::declval<const VectorFESType&>()));

      /// @brief Pressure field.
      using ScalarGridFunctionType =
        decltype(Rodin::PETSc::Variational::GridFunction(
          std::declval<const ScalarFESType&>()));

      /// @brief Velocity unknown.
      using VectorTrialFunctionType =
        decltype(Rodin::PETSc::Variational::TrialFunction(
          std::declval<const VectorFESType&>()));

      /// @brief Pressure unknown.
      using ScalarTrialFunctionType =
        decltype(Rodin::PETSc::Variational::TrialFunction(
          std::declval<const ScalarFESType&>()));

      /// @brief Velocity test function.
      using VectorTestFunctionType =
        decltype(Rodin::PETSc::Variational::TestFunction(
          std::declval<const VectorFESType&>()));

      /// @brief Pressure test function.
      using ScalarTestFunctionType =
        decltype(Rodin::PETSc::Variational::TestFunction(
          std::declval<const ScalarFESType&>()));

      /// @brief Coupled velocity and pressure problem.
      using FlowProblemType = decltype(Rodin::Variational::Problem(
        std::declval<VectorTrialFunctionType&>(),
        std::declval<ScalarTrialFunctionType&>(),
        std::declval<VectorTestFunctionType&>(),
        std::declval<ScalarTestFunctionType&>()));

      /// @brief Mass problem of the velocity space.
      using VectorProblemType = decltype(Rodin::Variational::Problem(
        std::declval<VectorTrialFunctionType&>(),
        std::declval<VectorTestFunctionType&>()));

      /// @brief Mass problem of the pressure space.
      using ScalarProblemType = decltype(Rodin::Variational::Problem(
        std::declval<ScalarTrialFunctionType&>(),
        std::declval<ScalarTestFunctionType&>()));

      /// @brief Solver of the flow problem.
      using FlowSolverType =
        decltype(Rodin::PETSc::Solver::KSP(std::declval<FlowProblemType&>()));

      /// @brief Solver of the velocity mass problem.
      using VectorSolverType =
        decltype(Rodin::PETSc::Solver::KSP(std::declval<VectorProblemType&>()));

      /// @brief Solver of the pressure mass problem.
      using ScalarSolverType =
        decltype(Rodin::PETSc::Solver::KSP(std::declval<ScalarProblemType&>()));

      /**
       * @brief Copies the mesh to the device and builds the problems.
       * @param[in] mesh Host mesh, which must outlive this object.
       * @param[in] cfg Configuration of the atrium.
       * @param[in] inlets Attributes of the pressure inlets.
       */
      AtriumDeviceFlow(const HostMeshType& mesh, const Config& cfg,
        const Rodin::FlatSet<AtriumBase::Attribute>& inlets)
        : m_cfg(cfg),
          m_inlets(inlets),
          m_mesh(ContextType(mesh.getContext()), mesh),
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
          m_piTilde(m_sh),
          m_convProjection(m_vh),
          m_sub(m_vh),
          m_subOld(m_vh),
          m_pIn(m_sh),
          m_pOut(m_sh),
          m_mitralZ(m_sh),
          m_flow(m_u, m_p, m_v, m_q),
          m_flowKSP(m_flow),
          m_scalarProjection(m_sTrial, m_sTest),
          m_scalarProjectionKSP(m_scalarProjection),
          m_vectorProjection(m_wTrial, m_wTest),
          m_vectorProjectionKSP(m_vectorProjection)
      {
        m_scalarProjectionKSP.setPrefix("atrium_sproj_");
        m_vectorProjectionKSP.setPrefix("atrium_vproj_");
        fill(m_uOld, 0);
        fill(m_uLin, 0);
        fill(m_subOld, 0);
        fill(m_sub, 0);
        fill(m_convProjection, 0);
        fill(m_piTilde, 0);
        setup();
      }

      AtriumDeviceFlow(const AtriumDeviceFlow&) = delete;

      AtriumDeviceFlow& operator=(const AtriumDeviceFlow&) = delete;

      /**
       * @brief Sets the velocity of the previous step.
       * @param[in] velocity Coefficients of the host velocity.
       */
      void setPrevious(::Vec velocity)
      {
        copy(velocity, m_uOld);
      }

      /**
       * @brief Sets the velocity the convective term is linearised about.
       * @param[in] velocity Coefficients of the host velocity.
       */
      void setIterate(::Vec velocity)
      {
        copy(velocity, m_uLin);
      }

      /**
       * @brief Sets the scalars of the openings for the current step.
       * @param[in] pIn Pressure of the inlets.
       * @param[in] pOut Pressure behind the outlet.
       * @param[in] mitralZ Resistance of the outlet times its area.
       */
      void setOpenings(Real pIn, Real pOut, Real mitralZ)
      {
        fill(m_pIn, pIn);
        fill(m_pOut, pOut);
        fill(m_mitralZ, mitralZ);
      }

      /// @brief Computes the orthogonal-subscale projections of the step.
      void project()
      {
        using namespace Rodin::Variational;
        const Real rho = m_cfg.rho;
        const Real dt = m_cfg.dt;
        const auto conv = Jacobian(m_uOld) * m_uOld;

        m_vectorProjection = Integral(m_wTrial, m_wTest) - Integral(conv, m_wTest);
        m_vectorProjection.solve(m_vectorProjectionKSP);
        copy(m_wTrial.getSolution().getData(), m_convProjection);

        m_vectorProjection = Integral(m_wTrial, m_wTest) -
          Integral(rho * vmsTau() * ((1.0 / dt) * m_subOld - conv + m_convProjection),
            m_wTest);
        m_vectorProjection.solve(m_vectorProjectionKSP);
        copy(m_wTrial.getSolution().getData(), m_sub);

        m_scalarProjection = Integral(m_sTrial, m_sTest) -
          Integral(sqrtTauC() * Div(m_uOld), m_sTest);
        m_scalarProjection.solve(m_scalarProjectionKSP);
        copy(m_sTrial.getSolution().getData(), m_piTilde);
      }

      /// @brief Assembles the flow system.
      void assemble()
      {
        m_flow.assemble();
        if (!m_fieldSplitsSet)
        {
          m_flow.setFieldSplits();
          m_fieldSplitsSet = true;
        }
      }

      /// @brief Solves the assembled flow system.
      void solve()
      {
        m_flow.solve(m_flowKSP);
        m_u.getSolution().sync();
        m_p.getSolution().sync();
      }

      /// @brief Keeps the subscale of the step for the next one.
      void advance()
      {
        copy(m_sub.getData(), m_subOld);
      }

      /// @brief Returns the solver of the flow system. @return PETSc handle.
      ::KSP getKSP()
      {
        return m_flowKSP.getHandle();
      }

      /// @brief Returns the solved velocity. @return Coefficient vector.
      ::Vec getVelocity()
      {
        return m_u.getSolution().getData();
      }

      /// @brief Returns the solved pressure. @return Coefficient vector.
      ::Vec getPressure()
      {
        return m_p.getSolution().getData();
      }

      auto& getProblem()
      {
        return m_flow;
      }

      /// @brief Returns the number of unknowns. @return Global size.
      size_t getSize() const
      {
        return m_vh.getSize() + m_sh.getSize();
      }

    private:
      /// @brief Copies a vector into a device field.
      template <class GridFunction>
      static void copy(::Vec from, GridFunction& to)
      {
        [[maybe_unused]] PetscErrorCode ierr = VecCopy(from, to.getData());
        assert(ierr == PETSC_SUCCESS);
        refresh(to);
      }

      /// @brief Sets every coefficient of a device field.
      template <class GridFunction>
      static void fill(GridFunction& to, Real value)
      {
        [[maybe_unused]] PetscErrorCode ierr = VecSet(to.getData(), value);
        assert(ierr == PETSC_SUCCESS);
        refresh(to);
      }

      /**
       * @brief Sends the owned coefficients of a device field to its ghosts.
       *
       * Writing through the PETSc handle only reaches the owned
       * coefficients, while the assembly reads the ghost ones on the cells
       * and faces next to the other ranks.
       */
      template <class GridFunction>
      static void refresh(GridFunction& field)
      {
        field.sync();
        [[maybe_unused]] PetscErrorCode ierr =
          VecGhostUpdateBegin(field.getData(), INSERT_VALUES, SCATTER_FORWARD);
        assert(ierr == PETSC_SUCCESS);
        ierr = VecGhostUpdateEnd(field.getData(), INSERT_VALUES, SCATTER_FORWARD);
        assert(ierr == PETSC_SUCCESS);
      }

      /// @brief Size of the cells as a function the device evaluates.
      static Rodin::Variational::RealFunction<AtriumCellSize> cellSize()
      {
        return Rodin::Variational::RealFunction(AtriumCellSize{});
      }

      /// @brief Carreau-Yasuda viscosity at the previous velocity.
      auto viscosity() const
      {
        using namespace Rodin::Variational;
        const auto& cy = m_cfg.viscosity;
        const auto sym = 0.5 * (Jacobian(m_uOld) + Transpose(Jacobian(m_uOld)));
        const auto shear = Sqrt(
          cy.gammaRegularization * cy.gammaRegularization + 2.0 * Dot(sym, sym));
        return cy.muInf +
          (cy.mu0 - cy.muInf) *
          Pow(1.0 + Pow(cy.lambda * shear, cy.yasuda), (cy.n - 1.0) / cy.yasuda);
      }

      /// @brief Momentum time scale of the cell at the previous velocity.
      auto tau1() const
      {
        using namespace Rodin::Variational;
        const auto h = cellSize();
        return 1.0 /
          ((4.0 / m_cfg.rho) * viscosity() / (h * h) +
            2.0 * Sqrt(Dot(m_uOld, m_uOld)) / h);
      }

      /// @brief Parameter of the convective stabilisation.
      auto vmsTau() const
      {
        return m_cfg.vmsScale / (m_cfg.rho / m_cfg.dt + m_cfg.rho / tau1());
      }

      /// @brief Square root of the grad-div parameter.
      auto sqrtTauC() const
      {
        using namespace Rodin::Variational;
        const auto h = cellSize();
        return Sqrt((0.25 * m_cfg.gradDivScale * m_cfg.rho) * (h * h) / tau1());
      }

      /// @brief Parameter of the pressure stabilisation.
      auto tauP() const
      {
        return (m_cfg.pspgScale / m_cfg.rho) * tau1();
      }

      /// @brief Assigns the flow problem, which is then only reassembled.
      void setup()
      {
        using namespace Rodin::Variational;
        const Real rho = m_cfg.rho;
        const Real dt = m_cfg.dt;
        const BoundaryNormal normal(m_mesh);
        const VectorFunction zero{Real(0), Real(0), Real(0)};

        const auto symU = 0.5 * (Jacobian(m_u) + Transpose(Jacobian(m_u)));
        const auto symV = 0.5 * (Jacobian(m_v) + Transpose(Jacobian(m_v)));
        const auto uNormal = Dot(normal, m_u) * normal;
        const auto sqrtC = sqrtTauC();

        const auto inletBackflow = (0.5 * rho * m_cfg.inletBackflowStabilization) *
          Max(Dot(m_uOld, normal), 0.0);
        const auto outletBackflow = (0.5 * rho * m_cfg.outletBackflowStabilization) *
          Max(-Dot(m_uOld, normal), 0.0);

        // The coefficients tau and mu are not polynomial, so the quadrature
        // order of the terms carrying them is a policy, and the device
        // inference differs from the host one. Each order below is the one
        // the host path uses, so that both paths assemble the same system:
        // the VMS integrators of VMSConvectionIntegrator.h use 4K - 2, 3K - 1,
        // 2K - 2 and K with tau counted as order 0, the plain host Integral
        // falls back to k_trial + k_test - 2 for a pair of derivatives
        // (PSPG, with a P1 pressure: 0) and to k_trial + k_test otherwise
        // (viscous: 2K). Inferred orders left the two P1 paths 1e-4 apart in
        // max|u| after 20 steps.
        auto vmsBilinear =
          Integral((rho * rho) * vmsTau() * Jacobian(m_u) * m_uOld, Jacobian(m_v) * m_uOld);
        vmsBilinear.setOrder(4 * K - 2);
        auto vmsLinear = Integral(
          (rho * rho) * vmsTau() * (m_convProjection + (1.0 / dt) * m_subOld),
          Jacobian(m_v) * m_uOld);
        vmsLinear.setOrder(3 * K - 1);
        auto gradDivBilinear = Integral(sqrtC * sqrtC * Div(m_u), Div(m_v));
        gradDivBilinear.setOrder(2 * K - 2);
        auto gradDivLinear = Integral(sqrtC * m_piTilde, Div(m_v));
        gradDivLinear.setOrder(K);
        auto viscous = Integral(2.0 * viscosity() * symU, symV);
        viscous.setOrder(2 * K);
        auto pspg = Integral(tauP() * Grad(m_p), Grad(m_q));
        pspg.setOrder(0);

        const auto oseen = Integral((rho / dt) * m_u, m_v) -
          Integral((rho / dt) * m_uOld, m_v)

          + Integral(rho * Jacobian(m_u) * m_uLin, m_v) +
          Integral((0.5 * rho) * Div(m_uOld) * m_u, m_v)

          // Dynamic orthogonal subscale, same form as the host path: the
          // residual (grad u) u^n - Pi is implicit, the linear part only
          // carries the history tau rho / dt u'^n of the previous step.
          + vmsBilinear - vmsLinear

          + gradDivBilinear - gradDivLinear

          + viscous - Integral(m_p, Div(m_v)) +
          Integral(Div(m_u), m_q) + Integral(m_cfg.pressurePenalty * m_p, m_q)

          + pspg

          + BoundaryIntegral(m_pIn * normal, m_v).over(m_inlets) +
          BoundaryIntegral(m_pOut * normal, m_v).over(m_cfg.outlet)

          + BoundaryIntegral(m_cfg.inletImpedance * uNormal, m_v).over(m_inlets) +
          BoundaryIntegral(m_mitralZ * uNormal, m_v).over(m_cfg.outlet)

          + BoundaryIntegral(m_cfg.inletTangentialDamping * (m_u - uNormal), m_v)
              .over(m_inlets)

          + BoundaryIntegral(inletBackflow * m_u, m_v).over(m_inlets) +
          BoundaryIntegral(outletBackflow * m_u, m_v).over(m_cfg.outlet)

          + DirichletBC(m_u, zero).on(m_cfg.wall);

        if (m_cfg.newton)
        {
          m_flow = oseen + Integral(rho * Jacobian(m_uLin) * m_u, m_v) -
            Integral(rho * Jacobian(m_uLin) * m_uLin, m_v);
        }
        else
        {
          m_flow = oseen;
        }
      }

      Config m_cfg;
      Rodin::FlatSet<AtriumBase::Attribute> m_inlets;
      MeshType m_mesh;
      VectorFESType m_vh;
      ScalarFESType m_sh;

      VectorTrialFunctionType m_u;
      ScalarTrialFunctionType m_p;
      VectorTestFunctionType m_v;
      ScalarTestFunctionType m_q;
      VectorGridFunctionType m_uOld;
      VectorGridFunctionType m_uLin;

      ScalarTrialFunctionType m_sTrial;
      ScalarTestFunctionType m_sTest;
      VectorTrialFunctionType m_wTrial;
      VectorTestFunctionType m_wTest;

      ScalarGridFunctionType m_piTilde;
      VectorGridFunctionType m_convProjection;
      VectorGridFunctionType m_sub;
      VectorGridFunctionType m_subOld;

      // Scalars of the openings, constant fields the assembled form reads.
      ScalarGridFunctionType m_pIn;
      ScalarGridFunctionType m_pOut;
      ScalarGridFunctionType m_mitralZ;

      FlowProblemType m_flow;
      FlowSolverType m_flowKSP;
      ScalarProblemType m_scalarProjection;
      ScalarSolverType m_scalarProjectionKSP;
      VectorProblemType m_vectorProjection;
      VectorSolverType m_vectorProjectionKSP;

      bool m_fieldSplitsSet = false;
  };
}

#endif
