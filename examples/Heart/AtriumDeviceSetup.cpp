/*
 *          Copyright Oscar RUZ NUNES 2026.
 * Distributed under the Boost Software License, Version 1.0.
 *       (See accompanying file LICENSE or copy at
 *          https://www.boost.org/LICENSE_1_0.txt)
 */
/**
 * @file AtriumDeviceSetup.cpp
 * @brief Construction of the device flow of Atrium<K>: spaces, fields and forms.
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
  void Atrium<K>::deviceCreate()
  {
    m_device = std::make_unique<AtriumDeviceFlow<K>>(m_mesh, m_cfg, m_inletSet);
  }

  template void Atrium<RODIN_EXAMPLES_ATRIUM_K>::deviceCreate();
}
