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


#include <3rdparty/miniexr/miniexr.h>

#define ARRAY_SIZE(x) sizeof(x)/sizeof(x[0])

// Writes EXR into a memory buffer.
// Input:
//   - (width) x (height) image,
//   - channels=4: 8 bytes per pixel (R,G,B,A order, 16 bit float per channel; alpha ignored), or
//   - channels=3: 6 bytes per pixel (R,G,B order, 16 bit float per channel).
// Returns memory buffer with .EXR contents and buffer size in outSize. free() the buffer when done with it.
unsigned char* miniexr_write(unsigned width, unsigned height, unsigned channels, const void* rgba16f, size_t* outSize)
{
	const unsigned ww = width - 1;
	const unsigned hh = height - 1;
	const unsigned char kHeader[] = {
		0x76, 0x2f, 0x31, 0x01, // magic
		2, 0, 0, 0, // version, scanline
		// channels
		'c', 'h', 'a', 'n', 'n', 'e', 'l', 's', 0,
		'c', 'h', 'l', 'i', 's', 't', 0,
		55, 0, 0, 0,
		'B', 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, // R, half
		'G', 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, // G, half
		'R', 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, // B, half
		0,
		// compression
		'c', 'o', 'm', 'p', 'r', 'e', 's', 's', 'i', 'o', 'n', 0,
		'c', 'o', 'm', 'p', 'r', 'e', 's', 's', 'i', 'o', 'n', 0,
		1, 0, 0, 0,
		0, // no compression
		// dataWindow
		'd', 'a', 't', 'a', 'W', 'i', 'n', 'd', 'o', 'w', 0,
		'b', 'o', 'x', '2', 'i', 0,
		16, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 0,
		ww & 0xFF, (ww >> 8) & 0xFF, (ww >> 16) & 0xFF, (ww >> 24) & 0xFF,
		hh & 0xFF, (hh >> 8) & 0xFF, (hh >> 16) & 0xFF, (hh >> 24) & 0xFF,
		// displayWindow
		'd', 'i', 's', 'p', 'l', 'a', 'y', 'W', 'i', 'n', 'd', 'o', 'w', 0,
		'b', 'o', 'x', '2', 'i', 0,
		16, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 0,
		ww & 0xFF, (ww >> 8) & 0xFF, (ww >> 16) & 0xFF, (ww >> 24) & 0xFF,
		hh & 0xFF, (hh >> 8) & 0xFF, (hh >> 16) & 0xFF, (hh >> 24) & 0xFF,
		// lineOrder
		'l', 'i', 'n', 'e', 'O', 'r', 'd', 'e', 'r', 0,
		'l', 'i', 'n', 'e', 'O', 'r', 'd', 'e', 'r', 0,
		1, 0, 0, 0,
		0, // increasing Y
		// pixelAspectRatio
		'p', 'i', 'x', 'e', 'l', 'A', 's', 'p', 'e', 'c', 't', 'R', 'a', 't', 'i', 'o', 0,
		'f', 'l', 'o', 'a', 't', 0,
		4, 0, 0, 0,
		0, 0, 0x80, 0x3f, // 1.0f
		// screenWindowCenter
		's', 'c', 'r', 'e', 'e', 'n', 'W', 'i', 'n', 'd', 'o', 'w', 'C', 'e', 'n', 't', 'e', 'r', 0,
		'v', '2', 'f', 0,
		8, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 0,
		// screenWindowWidth
		's', 'c', 'r', 'e', 'e', 'n', 'W', 'i', 'n', 'd', 'o', 'w', 'W', 'i', 'd', 't', 'h', 0,
		'f', 'l', 'o', 'a', 't', 0,
		4, 0, 0, 0,
		0, 0, 0x80, 0x3f, // 1.0f
		// end of header
		0,
	};
	const int kHeaderSize = ARRAY_SIZE(kHeader);

	const int kScanlineTableSize = 8 * height;
	const unsigned pixelRowSize = width * 3 * 2;
	const unsigned fullRowSize = pixelRowSize + 8;

	unsigned bufSize = kHeaderSize + kScanlineTableSize + height * fullRowSize;
	unsigned char* buf = (unsigned char*)malloc(bufSize);
	if (!buf)
		return NULL;

	// copy in header
	memcpy(buf, kHeader, kHeaderSize);

	// line offset table
	unsigned ofs = kHeaderSize + kScanlineTableSize;
	unsigned char* ptr = buf + kHeaderSize;
	for (unsigned int y = 0; y < height; ++y)
	{
		*ptr++ = ofs & 0xFF;
		*ptr++ = (ofs >> 8) & 0xFF;
		*ptr++ = (ofs >> 16) & 0xFF;
		*ptr++ = (ofs >> 24) & 0xFF;
		*ptr++ = 0;
		*ptr++ = 0;
		*ptr++ = 0;
		*ptr++ = 0;
		ofs += fullRowSize;
	}

	// scanline data
	const unsigned char* src = (const unsigned char*)rgba16f;
	const int stride = channels * 2;
	for (unsigned int y = 0; y < height; ++y)
	{
		// coordinate
		*ptr++ = y & 0xFF;
		*ptr++ = (y >> 8) & 0xFF;
		*ptr++ = (y >> 16) & 0xFF;
		*ptr++ = (y >> 24) & 0xFF;
		// data size
		*ptr++ = pixelRowSize & 0xFF;
		*ptr++ = (pixelRowSize >> 8) & 0xFF;
		*ptr++ = (pixelRowSize >> 16) & 0xFF;
		*ptr++ = (pixelRowSize >> 24) & 0xFF;
		// B, G, R
		memcpy(ptr, src, width * 6);
		const unsigned char* chsrc;
		chsrc = src + 4;
		for (unsigned int x = 0; x < width; ++x)
		{
			*ptr++ = chsrc[0];
			*ptr++ = chsrc[1];
			chsrc += stride;
		}
		chsrc = src + 2;
		for (unsigned int x = 0; x < width; ++x)
		{
			*ptr++ = chsrc[0];
			*ptr++ = chsrc[1];
			chsrc += stride;
		}
		chsrc = src + 0;
		for (unsigned int x = 0; x < width; ++x)
		{
			*ptr++ = chsrc[0];
			*ptr++ = chsrc[1];
			chsrc += stride;
		}

		src += width * stride;
	}

	assert(ptr - buf == bufSize);

	*outSize = bufSize;
	return buf;
}

unsigned short FloatToHalf(float f)
{
	union convertf2i {
		unsigned int i;
		float f;
	};
	convertf2i f2i;
	f2i.f = f;
	unsigned int i = f2i.i;

	if (i == 0)
		return 0;

	// Not robust at handling denormals, infinities, ...
	return ((i >> 16) & 0x8000) | ((((i & 0x7f800000) - 0x38000000) >> 13) & 0x7c00) | ((i >> 13) & 0x03ff);
}
