/*
 *          Copyright Oscar RUZ NUNES 2026.
 * Distributed under the Boost Software License, Version 1.0.
 *       (See accompanying file LICENSE or copy at
 *          https://www.boost.org/LICENSE_1_0.txt)
 */
/**
 * @file AtriumDeviceShear.cpp
 * @brief Wall shear stress and fluxes of the device flow of Atrium<K>.
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

  template void Atrium<RODIN_EXAMPLES_ATRIUM_K>::deviceWallShear();
  template void Atrium<RODIN_EXAMPLES_ATRIUM_K>::deviceFluxes();
  template void Atrium<RODIN_EXAMPLES_ATRIUM_K>::devicePullWallShear();
  template void Atrium<RODIN_EXAMPLES_ATRIUM_K>::devicePullCycle();
}
