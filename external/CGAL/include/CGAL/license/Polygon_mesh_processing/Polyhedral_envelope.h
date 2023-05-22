// Copyright (c) 2016  GeometryFactory SARL (France).
// All rights reserved.
//
// This file is part of CGAL (www.cgal.org)
//
// $URL: https://github.com/CGAL/cgal/blob/v5.3/Installation/include/CGAL/license/Polygon_mesh_processing/Polyhedral_envelope.h $
// $Id: Polyhedral_envelope.h 6b9bd51 2020-11-12T14:45:14+00:00 Andreas Fabri
// SPDX-License-Identifier: LGPL-3.0-or-later OR LicenseRef-Commercial
//
// Author(s) : Andreas Fabri
//
// Warning: this file is generated, see include/CGAL/licence/README.md

#ifndef CGAL_LICENSE_POLYGON_MESH_PROCESSING_POLYHEDRAL_ENVELOPE_H
#define CGAL_LICENSE_POLYGON_MESH_PROCESSING_POLYHEDRAL_ENVELOPE_H

#include <CGAL/config.h>
#include <CGAL/license.h>

#ifdef CGAL_POLYGON_MESH_PROCESSING_POLYHEDRAL_ENVELOPE_COMMERCIAL_LICENSE

#  if CGAL_POLYGON_MESH_PROCESSING_POLYHEDRAL_ENVELOPE_COMMERCIAL_LICENSE < CGAL_RELEASE_DATE

#    if defined(CGAL_LICENSE_WARNING)

       CGAL_pragma_warning("Your commercial license for CGAL does not cover "
                           "this release of the Polygon Mesh Processing - Polyhedral envelope package.")
#    endif

#    ifdef CGAL_LICENSE_ERROR
#      error "Your commercial license for CGAL does not cover this release \
              of the Polygon Mesh Processing - Polyhedral envelope package. \
              You get this error, as you defined CGAL_LICENSE_ERROR."
#    endif // CGAL_LICENSE_ERROR

#  endif // CGAL_POLYGON_MESH_PROCESSING_POLYHEDRAL_ENVELOPE_COMMERCIAL_LICENSE < CGAL_RELEASE_DATE

#else // no CGAL_POLYGON_MESH_PROCESSING_POLYHEDRAL_ENVELOPE_COMMERCIAL_LICENSE

#  if defined(CGAL_LICENSE_WARNING)
     CGAL_pragma_warning("\nThe macro CGAL_POLYGON_MESH_PROCESSING_POLYHEDRAL_ENVELOPE_COMMERCIAL_LICENSE is not defined."
                          "\nYou use the CGAL Polygon Mesh Processing - Polyhedral envelope package under "
                          "the terms of the GPLv3+.")
#  endif // CGAL_LICENSE_WARNING

#  ifdef CGAL_LICENSE_ERROR
#    error "The macro CGAL_POLYGON_MESH_PROCESSING_POLYHEDRAL_ENVELOPE_COMMERCIAL_LICENSE is not defined.\
            You use the CGAL Polygon Mesh Processing - Polyhedral envelope package under the terms of \
            the GPLv3+. You get this error, as you defined CGAL_LICENSE_ERROR."
#  endif // CGAL_LICENSE_ERROR

#endif // no CGAL_POLYGON_MESH_PROCESSING_ENVELOPE_COMMERCIAL_LICENSE

#endif // CGAL_LICENSE_POLYGON_MESH_PROCESSING_POLYHEDRAL_ENVELOPE_H