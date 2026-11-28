add_executable(CCMLC2014_0D CCMLC2014_0D.cpp)
target_link_libraries(CCMLC2014_0D
  PUBLIC
  Rodin::Heart
  Rodin::Alert
  )

add_executable(CoronarySolid CoronarySolid.cpp)
target_link_libraries(CoronarySolid
  PUBLIC
  Rodin::Solid
  Rodin::Heart
  Rodin::Solver
  Rodin::Geometry
  Rodin::Assembly
  Rodin::Variational
  )

if (RODIN_USE_PETSC AND RODIN_USE_MPI)
  add_executable(CoronaryArterySolid_PETSc_MPI CoronaryArterySolid_PETSc_MPI.cpp)
  target_link_libraries(CoronaryArterySolid_PETSc_MPI
    PUBLIC
    Rodin::Heart
    Rodin::Variational
    Rodin::PETSc
Rodin::MPI
    Rodin::Alert Rodin::Solver
    Rodin::Geometry
    Rodin::Assembly)
  if (RODIN_USE_SCOTCH)
    target_link_libraries(CoronaryArterySolid_PETSc_MPI PUBLIC Rodin::Scotch)
  endif()

  add_executable(CoronaryArtery_FSI_PETSc_MPI CoronaryArtery_FSI_PETSc_MPI.cpp)
  target_link_libraries(CoronaryArtery_FSI_PETSc_MPI
    PUBLIC
    Rodin::Heart
    Rodin::Variational
    Rodin::PETSc
    Rodin::MPI
    Rodin::Alert
    Rodin::Solver
    Rodin::Geometry
    Rodin::Assembly)
  if (RODIN_USE_SCOTCH)
    target_link_libraries(CoronaryArtery_FSI_PETSc_MPI PUBLIC Rodin::Scotch)
  endif()

  add_executable(CoronaryArtery_FSI_Explicit_PETSc_MPI CoronaryArtery_FSI_Explicit_PETSc_MPI.cpp)
  target_link_libraries(CoronaryArtery_FSI_Explicit_PETSc_MPI
    PUBLIC
    Rodin::Heart
    Rodin::Variational
    Rodin::PETSc
    Rodin::MPI
    Rodin::Alert
    Rodin::Solver
    Rodin::Geometry
    Rodin::Assembly)
  if (RODIN_USE_SCOTCH)
    target_link_libraries(CoronaryArtery_FSI_Explicit_PETSc_MPI PUBLIC Rodin::Scotch)
  endif()
endif()

# Sequential (single-process) explicit coronary FSI: PETSc only, no MPI.
if (RODIN_USE_PETSC)
  add_executable(CoronaryArtery_FSI_Explicit_PETSc_Seq CoronaryArtery_FSI_Explicit_PETSc_Seq.cpp)
  target_link_libraries(CoronaryArtery_FSI_Explicit_PETSc_Seq
    PUBLIC
    Rodin::Heart
    Rodin::Variational
    Rodin::PETSc
    Rodin::Alert
    Rodin::Solver
    Rodin::Geometry
    Rodin::Assembly)

  # Equal-order P1/P1 variant (separate results_p1p1/ output folder), so it
  # can run concurrently with the P2/P1 case on different cores.
  add_executable(CoronaryArtery_FSI_Explicit_P1P1_PETSc_Seq CoronaryArtery_FSI_Explicit_P1P1_PETSc_Seq.cpp)
  target_link_libraries(CoronaryArtery_FSI_Explicit_P1P1_PETSc_Seq
    PUBLIC
    Rodin::Heart
    Rodin::Variational
    Rodin::PETSc
    Rodin::Alert
    Rodin::Solver
    Rodin::Geometry
    Rodin::Assembly)

  # Coagulation-kinetics testbed on a simplified geometry: one inlet, one
  # outlet, an R_mu-Windkessel outlet and an ectatic segment that provides the
  # recirculation the kinetics need. Sequential and PETSc-only, like the P1/P1
  # case above; it does not link Rodin::MPI.
  #
  # The mesh ships in resources/examples/Heart/ProteinTest.mesh and is
  # regenerated with:
  #   python3 examples/Heart/ProteinTest/make_test_mesh.py \
  #           resources/examples/Heart/ProteinTest.mesh
  # The executable looks for it at ../resources/examples/Heart/ProteinTest.mesh
  # relative to the working directory, so run it from the build tree or pass a
  # different path in Config::meshPath.
  add_executable(ProteinTest ProteinTest.cpp)
  target_link_libraries(ProteinTest
    PUBLIC
    Rodin::Heart
    Rodin::Variational
    Rodin::PETSc
    Rodin::Alert
    Rodin::Solver
    Rodin::Geometry
    Rodin::Assembly)
  target_include_directories(ProteinTest
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
endif()

if (RODIN_USE_PETSC AND RODIN_USE_MPI)
  # Atrium: four pressure inlets, one outlet closed by the 0D LV through a
  # mitral diode, VMS/PSPG-stabilised P1/P1 flow and the coagulation kinetics.
  #
  # Atrium<1> and Atrium<2> are instantiated in one object library each, and
  # the device flow step is spread over several sources, so that the device
  # compiler works on ten small units in parallel instead of one: under NVCC
  # the single unit took hours. Each source instantiates the members it
  # defines for the degree RODIN_EXAMPLES_ATRIUM_K.
  set(ATRIUM_UNITS AtriumHost.cpp)
  set(ATRIUM_DEVICE OFF)
  if (TARGET Rodin::Kokkos::MPI AND TARGET Rodin::Kokkos::PETSc)
    set(ATRIUM_DEVICE ON)
    list(APPEND ATRIUM_UNITS
      AtriumDeviceSetup.cpp
      AtriumDeviceAssemble.cpp
      AtriumDeviceProject.cpp
      AtriumDeviceShear.cpp)
  endif()

  add_executable(Atrium Atrium.cpp)
  foreach(ATRIUM_K 1 2)
    add_library(AtriumP${ATRIUM_K} OBJECT ${ATRIUM_UNITS})
    target_compile_definitions(AtriumP${ATRIUM_K}
      PRIVATE RODIN_EXAMPLES_ATRIUM_K=${ATRIUM_K})
    target_link_libraries(Atrium PRIVATE AtriumP${ATRIUM_K})
  endforeach()

  foreach(ATRIUM_TARGET Atrium AtriumP1 AtriumP2)
    target_link_libraries(${ATRIUM_TARGET}
      PUBLIC
      Rodin::Heart
      Rodin::Variational
      Rodin::PETSc
      Rodin::MPI
      Rodin::Alert
      Rodin::Solver
      Rodin::Geometry
      Rodin::Assembly)
    target_include_directories(${ATRIUM_TARGET} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    if (RODIN_USE_SCOTCH)
      target_link_libraries(${ATRIUM_TARGET} PUBLIC Rodin::Scotch)
    endif()
    # The flow step runs on the Kokkos device under -atrium_device.
    if (ATRIUM_DEVICE)
      target_link_libraries(${ATRIUM_TARGET}
        PUBLIC Rodin::Kokkos::MPI Rodin::Kokkos::PETSc)
      target_compile_definitions(${ATRIUM_TARGET} PRIVATE RODIN_EXAMPLES_ATRIUM_DEVICE)
    endif()
  endforeach()

  add_executable(CoronaryArtery
    CoronaryArtery.cpp
    CoronaryArtery/CoupledLV0DCoronary3D.cpp
    )
  target_link_libraries(CoronaryArtery
    PUBLIC
    Rodin::Heart
    Rodin::Variational
    Rodin::PETSc
    Rodin::MPI
    Rodin::Alert
    )

  if (RODIN_USE_SCOTCH)
    target_link_libraries(CoronaryArtery PUBLIC Rodin::Scotch)
  endif()
endif()
