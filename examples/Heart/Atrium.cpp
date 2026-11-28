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
#include <cassert>
#include <iostream>
#include <stdexcept>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>

#include <petscsys.h>

#include <Rodin/Alert.h>
#include <Rodin/Configure.h>

#ifdef RODIN_EXAMPLES_ATRIUM_DEVICE
#include <Kokkos_Core.hpp>
#endif

#include "Atrium.h"

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
    // PCMG smooths its finest level with Amat by default, which for this
    // split is the implicit Schur complement: every smoother product then
    // runs a velocity V-cycle. Smoothing on the assembled selfp matrix
    // halves the solve time for the same iteration count.
    setPETScDefault("-fieldsplit_pressure_pc_use_amat", "false");
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
