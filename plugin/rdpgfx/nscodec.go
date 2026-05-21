// Package rdpgfx — NSCodec (MS-RDPNSC) decoder.
//
// This file is a faithful Go port of FreeRDP's libfreerdp/codec/nsc.c
// (the C reference lives at /tmp/freerdp-ref/nsc.c in this work tree).
// Each helper carries a `// FreeRDP: nsc.c line <N>` comment pointing
// back at the canonical implementation it mirrors.
//
// NSCodec is invoked from clearcodec.go's subcodec-1 branch.  The
// decoder writes BGRA32 pixels (B at byte 0, A=0xFF) into the caller's
// surface buffer at the requested (dstX, dstY) offset.

package rdpgfx

import (
	"bytes"
	"log/slog"

	"github.com/nakagami/grdp/core"
)

// nscDecoder owns the four plane buffers reused across rectangles so
// that the hot path doesn't reallocate on every invocation.  Buffer
// indices match FreeRDP's PlaneBuffers[] convention: 0=Y, 1=Co, 2=Cg,
// 3=A. FreeRDP: nsc_types.h line 45.
type nscDecoder struct {
	planes [4][]byte
}

// newNSCodec constructs an nscDecoder with empty plane buffers; the
// buffers grow on demand inside Decode().
func newNSCodec() *nscDecoder {
	return &nscDecoder{}
}

// roundUpTo mirrors ROUND_UP_TO from nsc_types.h line 35.  n must be a
// power of two.
func roundUpTo(b, n int) int {
	return (b + n - 1) &^ (n - 1)
}

// (We reuse `clampByte` from avc.go for MINMAX(_v, 0, 0xFF); FreeRDP
// nsc_types.h line 36.)

// Decode parses an NSCodec payload and writes a (width × height) BGRA32
// rectangle into out at (dstX, dstY).  outStride is in bytes (typically
// surfaceWidth * 4).  Returns false on any parse error; out is left in
// whatever partial state the decode reached, which is acceptable since
// the surrounding ClearCodec frame will overwrite the rectangle on the
// next refresh.
//
// FreeRDP: nsc_process_message + nsc_context_initialize + nsc_decode.
func (d *nscDecoder) Decode(data []byte, width, height int, out []byte, outStride, dstX, dstY int) bool {
	if width <= 0 || height <= 0 {
		return false
	}
	if dstX < 0 || dstY < 0 {
		return false
	}

	r := bytes.NewReader(data)

	// FreeRDP: nsc_stream_initialize (nsc.c line 233).  Header is 20
	// bytes: four uint32 plane byte counts, ColorLossLevel (u8),
	// ChromaSubsamplingLevel (u8), Reserved (u16).
	if r.Len() < 20 {
		slog.Debug("NSCodec: header truncated", "len", r.Len())
		return false
	}
	var planeByteCount [4]uint32
	var total uint64
	for i := 0; i < 4; i++ {
		v, _ := core.ReadUInt32LE(r)
		planeByteCount[i] = v
		total += uint64(v)
	}
	colorLossLevel, _ := core.ReadUInt8(r)
	chromaSubsampling, _ := core.ReadUInt8(r)
	_, _ = core.ReadUint16LE(r) // Reserved

	// FreeRDP: nsc.c lines 248-254 — ColorLossLevel must be in [1,7].
	if colorLossLevel < 1 || colorLossLevel > 7 {
		slog.Debug("NSCodec: ColorLossLevel out of range", "v", colorLossLevel)
		return false
	}

	if uint64(r.Len()) < total {
		slog.Debug("NSCodec: planes truncated",
			"have", r.Len(), "want", total)
		return false
	}

	// FreeRDP: nsc_context_initialize (nsc.c lines 262-313) — plane
	// dimensions depend on ChromaSubsamplingLevel.
	tempWidth := roundUpTo(width, 8)
	tempHeight := roundUpTo(height, 2)

	// orgByteCount[i] is the *decompressed* size of plane i.
	// FreeRDP: nsc.c lines 303-311.
	var orgByteCount [4]int
	for i := 0; i < 4; i++ {
		orgByteCount[i] = width * height
	}
	if chromaSubsampling != 0 {
		orgByteCount[0] = tempWidth * height           // Y
		orgByteCount[1] = (tempWidth >> 1) * (tempHeight >> 1) // Co
		orgByteCount[2] = orgByteCount[1]              // Cg
		// orgByteCount[3] (Alpha) stays width*height.
	}

	// Maximum plane buffer length, FreeRDP: nsc.c line 283.
	planeBufLen := tempWidth * tempHeight
	for i := 0; i < 4; i++ {
		need := orgByteCount[i]
		if need > planeBufLen {
			planeBufLen = need
		}
	}

	// Grow plane buffers if needed (reuse across calls).
	for i := 0; i < 4; i++ {
		if cap(d.planes[i]) < planeBufLen {
			d.planes[i] = make([]byte, planeBufLen)
		} else {
			d.planes[i] = d.planes[i][:planeBufLen]
		}
	}

	// FreeRDP: nsc_rle_decompress_data (nsc.c line 185).  Plane order
	// in the stream is the same as PlaneBuffers indices: Y, Co, Cg, A.
	// (FreeRDP iterates `i = 0..3`, reading from `context->Planes` and
	// writing into `priv->PlaneBuffers[i]`.)
	// Read the entire plane region first so we can hand each slice a
	// known size.
	planesBlob, _ := core.ReadBytes(int(total), r)
	if uint32(len(planesBlob)) < uint32(total) {
		return false
	}
	off := 0
	for i := 0; i < 4; i++ {
		ps := int(planeByteCount[i])
		os := orgByteCount[i]
		if ps == 0 {
			// FreeRDP: nsc.c lines 202-208 — missing plane fills with 0xFF.
			// For the alpha plane this means "opaque"; for chroma it
			// means "no chroma" (signed-cast to -1 after recovery, but
			// in practice the encoder only omits A in fully-opaque
			// frames).
			for j := 0; j < os; j++ {
				d.planes[i][j] = 0xFF
			}
		} else if ps < os {
			if !nscRLEDecode(planesBlob[off:off+ps], d.planes[i][:planeBufLen], os) {
				slog.Debug("NSCodec: RLE decode failed", "plane", i)
				return false
			}
		} else {
			// Plane is uncompressed; copy raw.
			// FreeRDP: nsc.c lines 216-224.
			if ps < os {
				return false
			}
			copy(d.planes[i][:os], planesBlob[off:off+os])
		}
		off += ps
	}

	// FreeRDP: nsc_decode (nsc.c line 44) — YCoCg + colorloss recovery
	// + optional chroma upsample → BGRA32 in one pass.
	shift := uint(colorLossLevel - 1)
	rw := tempWidth // rounded-up width used to index Y/Co/Cg planes when subsampled
	subsample := chromaSubsampling != 0

	yPlane := d.planes[0]
	coPlane := d.planes[1]
	cgPlane := d.planes[2]
	aPlane := d.planes[3]

	for y := 0; y < height; y++ {
		var yOff, cOff, aOff int
		if subsample {
			yOff = y * rw                // FreeRDP: nsc.c line 69
			cOff = (y >> 1) * (rw >> 1)  // FreeRDP: nsc.c line 70
		} else {
			yOff = y * width
			cOff = y * width
		}
		aOff = y * width

		dy := dstY + y
		// Start of destination row inside `out`.
		dstRow := dy*outStride + dstX*4

		// Bounds-check the row up-front; if it's outside the surface
		// we silently skip (matches FreeRDP's freerdp_image_copy clip).
		if dstRow < 0 || dstRow+width*4 > len(out) {
			continue
		}

		// Cache per-row plane base addresses (lets the inner loop be a
		// tight index-only sweep).
		yp := yPlane[yOff:]
		cop := coPlane[cOff:]
		cgp := cgPlane[cOff:]
		ap := aPlane[aOff:]

		for x := 0; x < width; x++ {
			// FreeRDP: nsc.c lines 82-84.  The double cast is load-
			// bearing: ((INT16)*plane) << shift truncates with sign-
			// extension to INT8, restoring the original signed
			// chroma.  In Go we replicate it explicitly with int8().
			yVal := int16(yp[x])
			var coIdx, cgIdx int
			if subsample {
				coIdx = x >> 1
				cgIdx = x >> 1
			} else {
				coIdx = x
				cgIdx = x
			}
			coVal := int16(int8(uint8(cop[coIdx]) << shift))
			cgVal := int16(int8(uint8(cgp[cgIdx]) << shift))

			// FreeRDP: nsc.c lines 85-87.
			rVal := yVal + coVal - cgVal
			gVal := yVal + cgVal
			bVal := yVal - coVal - cgVal

			di := dstRow + x*4
			// FreeRDP: nsc.c lines 93-96 — B, G, R, A order.
			out[di] = clampByte(int(bVal))
			out[di+1] = clampByte(int(gVal))
			out[di+2] = clampByte(int(rVal))
			out[di+3] = ap[x]
		}
	}

	return true
}

// nscRLEDecode is the byte-wise RLE inflater described in nsc.c.
//
// in:  compressed bytes (length inSize).
// out: target buffer; the function writes `originalSize` bytes into it.
//
// The encoding uses a literal-run scheme where two identical adjacent
// bytes signal a run; the byte following the duplicate gives the count
// (count + 2 bytes total) or, if 0xFF, introduces a uint32 LE extended
// count.  The trailing 4 bytes of the plane are always emitted
// verbatim.
//
// FreeRDP: nsc_rle_decode (nsc.c line 107).
func nscRLEDecode(in, out []byte, originalSize int) bool {
	if originalSize < 0 {
		return false
	}
	if originalSize == 0 {
		return true
	}
	// FreeRDP requires at least 4 trailing literal bytes; if the
	// original plane is smaller than that the encoder would have just
	// emitted it uncompressed (handled by the caller).
	if originalSize < 4 {
		return false
	}

	left := originalSize
	outSize := len(out)
	inPos := 0
	outPos := 0

	for left > 4 {
		if inPos >= len(in) {
			return false
		}
		value := in[inPos]
		inPos++

		if left == 5 {
			// FreeRDP: nsc.c lines 121-129 — single literal at the
			// boundary with the 4-byte tail.
			if outPos >= outSize {
				return false
			}
			out[outPos] = value
			outPos++
			left--
			continue
		}

		if inPos >= len(in) {
			return false
		}
		next := in[inPos]

		if value == next {
			// Run.  Skip the duplicate byte.
			inPos++
			if inPos >= len(in) {
				return false
			}
			var runLen uint32
			if in[inPos] < 0xFF {
				// FreeRDP: nsc.c lines 139-144.
				runLen = uint32(in[inPos]) + 2
				inPos++
			} else {
				// FreeRDP: nsc.c lines 145-155 — extended uint32 LE
				// count.  Skip the 0xFF marker, then read 4 bytes.
				if inPos+5 > len(in) {
					return false
				}
				inPos++ // skip 0xFF
				runLen = uint32(in[inPos]) |
					uint32(in[inPos+1])<<8 |
					uint32(in[inPos+2])<<16 |
					uint32(in[inPos+3])<<24
				inPos += 4
			}
			if uint64(runLen) > uint64(outSize-outPos) || uint64(runLen) > uint64(left) {
				return false
			}
			for k := uint32(0); k < runLen; k++ {
				out[outPos] = value
				outPos++
			}
			left -= int(runLen)
		} else {
			// FreeRDP: nsc.c lines 166-173 — plain literal.
			if outPos >= outSize {
				return false
			}
			out[outPos] = value
			outPos++
			left--
		}
	}

	// FreeRDP: nsc.c lines 176-182 — copy the trailing 4 raw bytes.
	if outPos+4 > outSize || left < 4 {
		return false
	}
	if inPos+4 > len(in) {
		return false
	}
	copy(out[outPos:outPos+4], in[inPos:inPos+4])
	return true
}
