// miniexr.cpp - v0.2 - public domain - 2013 Aras Pranckevicius / Unity Technologies
//
// Writes OpenEXR RGB files out of half-precision RGBA or RGB data.
//
// Only tested on Windows (VS2008) and Mac (clang 3.3), little endian.
// Testing status: "works for me".
//
// History:
// 0.2 Source data can be RGB or RGBA now.
// 0.1 Initial release.

#ifndef MINIEXR_H
#define MINIEXR_H

#include <assert.h>
#include <string.h>
#include <stdlib.h>

unsigned char* miniexr_write(unsigned width, unsigned height, unsigned channels, const void* rgba16f, size_t* outSize);

unsigned short FloatToHalf(float f);

#endif
