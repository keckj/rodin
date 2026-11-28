/*
 *          Copyright Oscar RUZ NUNES 2026.
 * Distributed under the Boost Software License, Version 1.0.
 *       (See accompanying file LICENSE or copy at
 *          https://www.boost.org/LICENSE_1_0.txt)
 */
/**
 * @file AtriumDeviceAssemble.cpp
 * @brief Assembly and solve of the device flow of Atrium<K>.
 *
 * Compiled once per velocity degree, which RODIN_EXAMPLES_ATRIUM_K sets
 * (examples/Heart/CMakeLists.txt): the device compiler then works on
 * smaller units, in parallel.
 */
#include <cassert>
#include <memory>

#include <petscvec.h>

#include "Atrium.h"
#include "AtriumDevice.h"

#ifndef RODIN_EXAMPLES_ATRIUM_K
#error "RODIN_EXAMPLES_ATRIUM_K, the velocity degree to instantiate, is not defined."
#endif

namespace Rodin::Examples::Heart
{
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

  template void Atrium<RODIN_EXAMPLES_ATRIUM_K>::deviceAssemble();
  template ::KSP Atrium<RODIN_EXAMPLES_ATRIUM_K>::deviceSolve();
  template ::KSP Atrium<RODIN_EXAMPLES_ATRIUM_K>::deviceKSP();
}
