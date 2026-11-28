/*
 *          Copyright Oscar RUZ NUNES 2026.
 * Distributed under the Boost Software License, Version 1.0.
 *       (See accompanying file LICENSE or copy at
 *          https://www.boost.org/LICENSE_1_0.txt)
 */
/**
 * @file AtriumDeviceProject.cpp
 * @brief Subscale projections of the device flow of Atrium<K>.
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
  void Atrium<K>::deviceEnd()
  {
    if (m_cfg.useVMS)
      m_device->advance();
  }

  template void Atrium<RODIN_EXAMPLES_ATRIUM_K>::deviceBegin(int);
  template void Atrium<RODIN_EXAMPLES_ATRIUM_K>::deviceEnd();
}
