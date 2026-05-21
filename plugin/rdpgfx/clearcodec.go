// Package rdpgfx — ClearCodec (MS-RDPEGFX 2.2.4) decoder.
//
// This file is a faithful Go port of FreeRDP's libfreerdp/codec/clear.c
// (the C reference lives at /tmp/freerdp-ref/clear.c in this work tree).
// Each helper carries a `// FreeRDP: <function> line <N>` comment
// pointing back at the canonical implementation it mirrors, so we can
// keep this port honest as FreeRDP evolves.
//
// All pixel buffers in this file are BGRA32, top-down, row-major.  The
// surface buffer (passed in as `out`) and every cache entry use that
// layout so we never need a colour-conversion pass — the C code
// internally calls convert_color() to flip between clear->format
// (BGRX32) and the destination format; here both ends are already
// BGRA32, so we copy directly and stamp alpha=0xFF.

package rdpgfx

import (
	"bytes"
	"log/slog"

	"github.com/nakagami/grdp/core"
)

// Header flag bits from MS-RDPEGFX 2.2.4.1 (CLEARCODEC_BITMAP_STREAM).
// FreeRDP: clear.c lines 34-36.
const (
	clearcodecFlagGlyphIndex uint8 = 0x01
	clearcodecFlagGlyphHit   uint8 = 0x02
	clearcodecFlagCacheReset uint8 = 0x04
)

// Cache sizes from FreeRDP: clear.c lines 38-39 and 64-68.
const (
	clearcodecGlyphCacheSize    = 4000
	clearcodecVBarSize          = 32768
	clearcodecVBarShortSize     = 16384
	clearcodecMaxVBarPixelCount = 52 // FreeRDP: clear.c line 679 and 720
)

// clearGlyphEntry mirrors CLEAR_GLYPH_ENTRY from clear.c line 41-46.
// pixels holds nWidth*nHeight BGRA32 pixels.
type clearGlyphEntry struct {
	width, height int
	pixels        []byte
}

// clearVBarEntry mirrors CLEAR_VBAR_ENTRY from clear.c line 48-53.
// pixels holds count BGRA32 pixels (one column of `count` rows).
type clearVBarEntry struct {
	count  int
	pixels []byte
}

// clearCodecCtx is the Go analogue of struct S_CLEAR_CONTEXT (clear.c line 55).
// It is created by newClearCodecCtx() and stored on GfxHandler.clearCtx;
// the decoder is reset (caches included) by recreating it on RESET_GRAPHICS,
// matching FreeRDP semantics that the caches survive between non-reset frames
// but state should be fresh after a server reset.
type clearCodecCtx struct {
	seqNumber             uint8
	seqInitialised        bool
	glyphCache            [clearcodecGlyphCacheSize]clearGlyphEntry
	vBarStorage           [clearcodecVBarSize]clearVBarEntry
	shortVBarStorage      [clearcodecVBarShortSize]clearVBarEntry
	vBarStorageCursor     int
	shortVBarStorageCursor int
	// nsc is the NSCodec sub-decoder used by subcodec id 1.  Held on
	// the context so its plane buffers are reused across rectangles
	// (and across frames) instead of being reallocated per call.
	nsc *nscDecoder
}

func newClearCodecCtx() *clearCodecCtx {
	return &clearCodecCtx{
		nsc: newNSCodec(),
	}
}

// resetVBarStorage mirrors clear_reset_vbar_storage (clear.c line 85).
// FreeRDP passes a `zero` flag to decide whether to free entries; in Go
// we just reset the cursors.  Cache-reset frames keep the entries alive
// (FreeRDP calls this with zero=FALSE in clear_decompress).
func (ctx *clearCodecCtx) resetVBarStorage(zero bool) {
	if zero {
		for i := range ctx.vBarStorage {
			ctx.vBarStorage[i] = clearVBarEntry{}
		}
		for i := range ctx.shortVBarStorage {
			ctx.shortVBarStorage[i] = clearVBarEntry{}
		}
	}
	ctx.vBarStorageCursor = 0
	ctx.shortVBarStorageCursor = 0
}

// decode is the public entry point invoked from rdpgfx.go's WTS1 dispatch
// (`g.clearCtx.decode(bmpData, w, h)`).  `data` is the full ClearCodec
// payload starting at glyphFlags; (w,h) is the bitmap rectangle size.
//
// The returned buffer is freshly allocated BGRA32 top-down of size w*h*4
// and is NOT borrowed from bitmapBufPool — the caller passes owned=false.
// On any parse error we return whatever was decoded so far (or a black
// buffer); we never panic.
//
// FreeRDP: clear_decompress (clear.c line 1078).
func (ctx *clearCodecCtx) decode(data []byte, w, h int) []byte {
	out := make([]byte, w*h*4)
	if w <= 0 || h <= 0 {
		return out
	}
	if len(data) < 2 {
		return out
	}

	r := bytes.NewReader(data)

	glyphFlags, _ := core.ReadUInt8(r)
	seqNumber, _ := core.ReadUInt8(r)

	// FreeRDP: clear.c lines 1130-1142.  Warn on mismatch but proceed.
	if !ctx.seqInitialised {
		ctx.seqNumber = seqNumber
		ctx.seqInitialised = true
	} else if seqNumber != ctx.seqNumber {
		slog.Debug("ClearCodec: unexpected sequence number",
			"got", seqNumber, "expected", ctx.seqNumber)
	}
	ctx.seqNumber = seqNumber + 1 // uint8 wraps naturally

	// FreeRDP: clear.c line 1144.  CACHE_RESET clears VBar cursors
	// (zero=FALSE keeps allocations alive across resets).
	if glyphFlags&clearcodecFlagCacheReset != 0 {
		ctx.resetVBarStorage(false)
	}

	// Glyph layer — sets glyphTarget to a cache slot we should populate
	// after composition completes (CLEARCODEC_FLAG_GLYPH_INDEX without HIT).
	// A non-nil cacheStore means "after decoding, copy `out` into
	// ctx.glyphCache[glyphIndex] so the next GLYPH_HIT can reuse it".
	// FreeRDP: clear_decompress_glyph_data (clear.c line 929).
	glyphCacheIndex, cacheAfter, ok := ctx.decompressGlyphData(r, glyphFlags, w, h, out)
	if !ok {
		return out
	}

	// If the frame is glyph-only (HIT+INDEX both set, no composition
	// payload), FreeRDP returns success here — the caller has the
	// glyph blitted into pDstData already.
	// FreeRDP: clear.c lines 1158-1170.
	if r.Len() < 12 {
		const mask = clearcodecFlagGlyphHit | clearcodecFlagGlyphIndex
		if glyphFlags&mask == mask {
			return out
		}
		// FreeRDP would treat this as an error; we just return what we have.
		return out
	}

	residualByteCount, _ := core.ReadUInt32LE(r)
	bandsByteCount, _ := core.ReadUInt32LE(r)
	subcodecByteCount, _ := core.ReadUInt32LE(r)

	if residualByteCount > 0 {
		if !decompressResidualData(r, residualByteCount, w, h, out) {
			return out
		}
	}
	if bandsByteCount > 0 {
		if !ctx.decompressBandsData(r, bandsByteCount, w, h, out) {
			return out
		}
	}
	if subcodecByteCount > 0 {
		if !ctx.decompressSubcodecsData(r, subcodecByteCount, w, h, out) {
			return out
		}
	}

	// After composition, copy the final framebuffer into the glyph cache
	// slot if this frame was a GLYPH_INDEX miss.  FreeRDP does this by
	// having clear_decompress_glyph_data hand out a pointer into the
	// cache so layers write into it directly; we do it as a post-copy.
	// FreeRDP: clear.c lines 1208-1264.
	if cacheAfter {
		ctx.storeGlyph(glyphCacheIndex, w, h, out)
	}

	return out
}

// decompressGlyphData mirrors clear_decompress_glyph_data (clear.c line 929).
//
// Returns (glyphIndex, cacheAfter, ok).  When cacheAfter is true the caller
// should copy the final composed frame into ctx.glyphCache[glyphIndex] once
// the residual/bands/subcodec layers have all finished.  When GLYPH_HIT is
// set the cached glyph is blitted into `out` immediately.
func (ctx *clearCodecCtx) decompressGlyphData(r *bytes.Reader, glyphFlags uint8, w, h int, out []byte) (uint16, bool, bool) {
	// FreeRDP: clear.c lines 942-946 — HIT without INDEX is an error.
	if glyphFlags&clearcodecFlagGlyphHit != 0 && glyphFlags&clearcodecFlagGlyphIndex == 0 {
		slog.Debug("ClearCodec: invalid glyph flags", "flags", glyphFlags)
		return 0, false, false
	}

	if glyphFlags&clearcodecFlagGlyphIndex == 0 {
		return 0, false, true
	}

	// FreeRDP: clear.c lines 951-956 — sanity limit.
	if w*h > 1024*1024 {
		slog.Debug("ClearCodec: glyph too large", "w", w, "h", h)
		return 0, false, false
	}

	if r.Len() < 2 {
		return 0, false, false
	}
	glyphIndex, _ := core.ReadUint16LE(r)
	if glyphIndex >= clearcodecGlyphCacheSize {
		slog.Debug("ClearCodec: invalid glyphIndex", "idx", glyphIndex)
		return 0, false, false
	}

	if glyphFlags&clearcodecFlagGlyphHit != 0 {
		// FreeRDP: clear.c lines 969-1002 — copy cached glyph into output.
		entry := &ctx.glyphCache[glyphIndex]
		if entry.pixels == nil || entry.width != w || entry.height != h {
			slog.Debug("ClearCodec: empty/mismatched glyph cache slot",
				"idx", glyphIndex, "want", []int{w, h},
				"have", []int{entry.width, entry.height})
			return glyphIndex, false, false
		}
		n := w * h * 4
		if n > len(entry.pixels) || n > len(out) {
			return glyphIndex, false, false
		}
		copy(out[:n], entry.pixels[:n])
		return glyphIndex, false, true
	}

	// GLYPH_INDEX without HIT: signal that we should store `out` after
	// the rest of the layers have finished composing.
	return glyphIndex, true, true
}

// storeGlyph copies the freshly-decoded frame into the glyph cache.
// FreeRDP equivalent is the allocation block in clear_decompress_glyph_data
// (clear.c lines 1004-1046) combined with all layers writing through the
// returned pointer; we do it as a single post-decode copy.
func (ctx *clearCodecCtx) storeGlyph(glyphIndex uint16, w, h int, out []byte) {
	if int(glyphIndex) >= clearcodecGlyphCacheSize {
		return
	}
	n := w * h * 4
	if n > len(out) {
		return
	}
	entry := &ctx.glyphCache[glyphIndex]
	if cap(entry.pixels) < n {
		entry.pixels = make([]byte, n)
	} else {
		entry.pixels = entry.pixels[:n]
	}
	copy(entry.pixels, out[:n])
	entry.width = w
	entry.height = h
}

// decompressResidualData decodes the RESIDUAL layer (RLE over BGR triples).
//
// FreeRDP: clear_decompress_residual_data (clear.c line 361).
func decompressResidualData(r *bytes.Reader, byteCount uint32, w, h int, out []byte) bool {
	if uint32(r.Len()) < byteCount {
		return false
	}

	pixelCount := uint32(w * h)
	var pixelIndex uint32
	var suboffset uint32

	for suboffset < byteCount {
		if r.Len() < 4 {
			return false
		}
		b, _ := core.ReadUInt8(r)
		g, _ := core.ReadUInt8(r)
		red, _ := core.ReadUInt8(r)
		runByte, _ := core.ReadUInt8(r)
		suboffset += 4

		runLengthFactor := uint32(runByte)
		// FreeRDP: clear.c lines 405-421 — extended run length encoding.
		if runLengthFactor >= 0xFF {
			if r.Len() < 2 {
				return false
			}
			rl16, _ := core.ReadUint16LE(r)
			runLengthFactor = uint32(rl16)
			suboffset += 2
			if runLengthFactor >= 0xFFFF {
				if r.Len() < 4 {
					return false
				}
				rl32, _ := core.ReadUInt32LE(r)
				runLengthFactor = rl32
				suboffset += 4
			}
		}

		if pixelIndex >= pixelCount || runLengthFactor > pixelCount-pixelIndex {
			slog.Debug("ClearCodec: residual run overflow",
				"pixelIndex", pixelIndex,
				"runLengthFactor", runLengthFactor,
				"pixelCount", pixelCount)
			return false
		}

		// FreeRDP writes into clear->TempBuffer in row-major BGRX32 then
		// calls convert_color() to copy to pDstData; for us out IS the
		// destination, so we paint runs row-by-row directly.
		for i := uint32(0); i < runLengthFactor; i++ {
			di := int(pixelIndex+i) * 4
			if di+4 > len(out) {
				break
			}
			out[di] = b
			out[di+1] = g
			out[di+2] = red
			out[di+3] = 0xFF
		}
		pixelIndex += runLengthFactor
	}

	if pixelIndex != pixelCount {
		slog.Debug("ClearCodec: residual pixelIndex != pixelCount",
			"pixelIndex", pixelIndex, "pixelCount", pixelCount)
		return false
	}
	return true
}

// decompressBandsData decodes the BANDS layer (VBar cache + short-VBar
// cache + colorBkg fill).
//
// This is the layer that was most broken in the legacy Go decoder:
// field order was (xStart, yStart, xEnd, yEnd) instead of FreeRDP's
// (xStart, xEnd, yStart, yEnd), and the cache-hit/miss bit patterns
// were inverted.  This port follows clear.c line-by-line.
//
// FreeRDP: clear_decompress_bands_data (clear.c line 606).
func (ctx *clearCodecCtx) decompressBandsData(r *bytes.Reader, byteCount uint32, surfW, surfH int, out []byte) bool {
	if uint32(r.Len()) < byteCount {
		return false
	}

	var suboffset uint32

	for suboffset < byteCount {
		if r.Len() < 11 {
			return false
		}
		// FreeRDP: clear.c lines 638-644 — note field order!
		xStart, _ := core.ReadUint16LE(r)
		xEnd, _ := core.ReadUint16LE(r)
		yStart, _ := core.ReadUint16LE(r)
		yEnd, _ := core.ReadUint16LE(r)
		cb, _ := core.ReadUInt8(r)
		cg, _ := core.ReadUInt8(r)
		cr, _ := core.ReadUInt8(r)
		suboffset += 11

		if xEnd < xStart || yEnd < yStart {
			slog.Debug("ClearCodec: bad band rectangle",
				"xStart", xStart, "xEnd", xEnd, "yStart", yStart, "yEnd", yEnd)
			return false
		}

		// FreeRDP: clear.c line 662 — inclusive width.
		vBarCount := uint32(xEnd-xStart) + 1
		vBarHeight := uint32(yEnd-yStart) + 1

		// FreeRDP: clear.c lines 679-682 — vBarHeight cap.
		if vBarHeight > 52 {
			slog.Debug("ClearCodec: vBarHeight > 52", "vBarHeight", vBarHeight)
			return false
		}

		for i := uint32(0); i < vBarCount; i++ {
			if r.Len() < 2 {
				return false
			}
			vBarHeader, _ := core.ReadUint16LE(r)
			suboffset += 2

			var vBarYOn uint16
			var vBarShortPixelCount uint32
			var vBarShortEntry *clearVBarEntry
			vBarUpdate := false

			switch {
			// FreeRDP: clear.c line 685 — SHORT_VBAR_CACHE_HIT.
			case (vBarHeader & 0xC000) == 0x4000:
				vBarIndex := vBarHeader & 0x3FFF
				if int(vBarIndex) >= len(ctx.shortVBarStorage) {
					slog.Debug("ClearCodec: short vBar idx out of range", "idx", vBarIndex)
					return false
				}
				vBarShortEntry = &ctx.shortVBarStorage[vBarIndex]
				if r.Len() < 1 {
					return false
				}
				y8, _ := core.ReadUInt8(r)
				vBarYOn = uint16(y8)
				suboffset++
				vBarShortPixelCount = uint32(vBarShortEntry.count)
				vBarUpdate = true

			// FreeRDP: clear.c line 705 — SHORT_VBAR_CACHE_MISS.
			case (vBarHeader & 0xC000) == 0x0000:
				vBarYOn = vBarHeader & 0xFF
				vBarYOff := (vBarHeader >> 8) & 0x3F
				if vBarYOff < vBarYOn {
					slog.Debug("ClearCodec: vBarYOff < vBarYOn",
						"yOff", vBarYOff, "yOn", vBarYOn)
					return false
				}
				vBarShortPixelCount = uint32(vBarYOff - vBarYOn)
				if vBarShortPixelCount > 52 {
					slog.Debug("ClearCodec: vBarShortPixelCount > 52",
						"count", vBarShortPixelCount)
					return false
				}
				if uint32(r.Len()) < vBarShortPixelCount*3 {
					return false
				}

				// FreeRDP: clear.c line 730 — cursor must not overflow.
				if ctx.shortVBarStorageCursor >= clearcodecVBarShortSize {
					slog.Debug("ClearCodec: shortVBarStorageCursor overflow")
					return false
				}

				// Allocate / reuse the cache slot.
				vBarShortEntry = &ctx.shortVBarStorage[ctx.shortVBarStorageCursor]
				vBarShortEntry.count = int(vBarShortPixelCount)
				need := int(vBarShortPixelCount) * 4
				if cap(vBarShortEntry.pixels) < need {
					vBarShortEntry.pixels = make([]byte, need)
				} else {
					vBarShortEntry.pixels = vBarShortEntry.pixels[:need]
				}

				// Read BGR triples and pack as BGRA32.
				// FreeRDP: clear.c lines 745-760.
				for y := uint32(0); y < vBarShortPixelCount; y++ {
					b, _ := core.ReadUInt8(r)
					g, _ := core.ReadUInt8(r)
					red, _ := core.ReadUInt8(r)
					off := int(y) * 4
					vBarShortEntry.pixels[off] = b
					vBarShortEntry.pixels[off+1] = g
					vBarShortEntry.pixels[off+2] = red
					vBarShortEntry.pixels[off+3] = 0xFF
				}
				suboffset += vBarShortPixelCount * 3
				ctx.shortVBarStorageCursor = (ctx.shortVBarStorageCursor + 1) % clearcodecVBarShortSize
				vBarUpdate = true

			// FreeRDP: clear.c line 767 — VBAR_CACHE_HIT.
			case (vBarHeader & 0x8000) == 0x8000:
				vBarIndex := vBarHeader & 0x7FFF
				if int(vBarIndex) >= len(ctx.vBarStorage) {
					slog.Debug("ClearCodec: vBar idx out of range", "idx", vBarIndex)
					return false
				}
				entry := &ctx.vBarStorage[vBarIndex]
				// FreeRDP: clear.c lines 772-781 — fill dummy on empty.
				if entry.count == 0 {
					slog.Debug("ClearCodec: empty vBar cache slot, dummy data",
						"idx", vBarIndex)
					entry.count = int(vBarHeight)
					need := int(vBarHeight) * 4
					if cap(entry.pixels) < need {
						entry.pixels = make([]byte, need)
					} else {
						entry.pixels = entry.pixels[:need]
						for i := range entry.pixels {
							entry.pixels[i] = 0
						}
					}
				}

			// Default branch = VBAR_CACHE_MISS would land here in FreeRDP,
			// but its predecessors exhaust the bit space (0x4000/0x0000/
			// 0x8000-with-0x8000 set) so the only remaining patterns
			// would be the "0x8000 set + 0x4000 set" (0xC000) — already
			// matched as SHORT_HIT.  FreeRDP treats anything else as an
			// error; we mirror that.
			//
			// FreeRDP: clear.c line 783-788.
			default:
				slog.Debug("ClearCodec: invalid vBarHeader", "header", vBarHeader)
				return false
			}

			// FreeRDP: clear.c lines 790-879 — compose new vBar entry from
			// background + short pixels + background tail.
			var vBarEntry *clearVBarEntry
			if vBarUpdate {
				if ctx.vBarStorageCursor >= clearcodecVBarSize {
					slog.Debug("ClearCodec: vBarStorageCursor overflow")
					return false
				}
				vBarEntry = &ctx.vBarStorage[ctx.vBarStorageCursor]
				vBarEntry.count = int(vBarHeight)
				need := int(vBarHeight) * 4
				if cap(vBarEntry.pixels) < need {
					vBarEntry.pixels = make([]byte, need)
				} else {
					vBarEntry.pixels = vBarEntry.pixels[:need]
				}

				vBarPixelCount := uint32(vBarHeight)

				// Region 1: rows [0, vBarYOn) ⇒ colorBkg.
				// FreeRDP: clear.c lines 813-826.
				y := uint32(0)
				count := uint32(vBarYOn)
				if y+count > vBarPixelCount {
					if vBarPixelCount > y {
						count = vBarPixelCount - y
					} else {
						count = 0
					}
				}
				for k := uint32(0); k < count; k++ {
					off := int(y+k) * 4
					vBarEntry.pixels[off] = cb
					vBarEntry.pixels[off+1] = cg
					vBarEntry.pixels[off+2] = cr
					vBarEntry.pixels[off+3] = 0xFF
				}

				// Region 2: rows [vBarYOn, vBarYOn+vBarShortPixelCount)
				// ⇒ vBarShortEntry pixels.
				// FreeRDP: clear.c lines 832-860.
				y = uint32(vBarYOn)
				count = vBarShortPixelCount
				if y+count > vBarPixelCount {
					if vBarPixelCount > y {
						count = vBarPixelCount - y
					} else {
						count = 0
					}
				}
				if count > 0 && vBarShortEntry != nil {
					srcOff := (int(y) - int(vBarYOn)) * 4
					if srcOff+int(count)*4 > len(vBarShortEntry.pixels) {
						slog.Debug("ClearCodec: shortVBar out of range")
						return false
					}
					for k := uint32(0); k < count; k++ {
						sOff := srcOff + int(k)*4
						dOff := (int(y) + int(k)) * 4
						vBarEntry.pixels[dOff] = vBarShortEntry.pixels[sOff]
						vBarEntry.pixels[dOff+1] = vBarShortEntry.pixels[sOff+1]
						vBarEntry.pixels[dOff+2] = vBarShortEntry.pixels[sOff+2]
						vBarEntry.pixels[dOff+3] = 0xFF
					}
				}

				// Region 3: rows [vBarYOn+vBarShortPixelCount, end) ⇒ colorBkg.
				// FreeRDP: clear.c lines 862-875.
				y = uint32(vBarYOn) + vBarShortPixelCount
				if vBarPixelCount > y {
					count = vBarPixelCount - y
				} else {
					count = 0
				}
				for k := uint32(0); k < count; k++ {
					off := (int(y) + int(k)) * 4
					vBarEntry.pixels[off] = cb
					vBarEntry.pixels[off+1] = cg
					vBarEntry.pixels[off+2] = cr
					vBarEntry.pixels[off+3] = 0xFF
				}

				ctx.vBarStorageCursor = (ctx.vBarStorageCursor + 1) % clearcodecVBarSize
			} else {
				// VBAR_CACHE_HIT branch — vBarEntry was set inline above.
				vBarIndex := vBarHeader & 0x7FFF
				vBarEntry = &ctx.vBarStorage[vBarIndex]
			}

			// FreeRDP: clear.c lines 881-890 — sanity check size.
			if vBarEntry.count != int(vBarHeight) {
				// FreeRDP: re-resize and continue; we do the same.
				slog.Debug("ClearCodec: vBarEntry count mismatch, resizing",
					"have", vBarEntry.count, "want", vBarHeight)
				vBarEntry.count = int(vBarHeight)
				need := int(vBarHeight) * 4
				if cap(vBarEntry.pixels) < need {
					vBarEntry.pixels = make([]byte, need)
				} else {
					vBarEntry.pixels = vBarEntry.pixels[:need]
				}
			}

			// FreeRDP: clear.c lines 892-922 — blit vBar column into surface.
			// nXDst/nYDst is 0 in our caller (the bitmap rectangle is the
			// entire decode output), so nXDstRel = xStart, nYDstRel = yStart.
			dx := int(xStart) + int(i)
			if dx >= surfW {
				return false
			}
			ycount := vBarEntry.count
			if ycount > surfH-int(yStart) {
				ycount = surfH - int(yStart)
			}
			for y := 0; y < ycount; y++ {
				dy := int(yStart) + y
				if dy >= surfH {
					return false
				}
				srcOff := y * 4
				if srcOff+4 > len(vBarEntry.pixels) {
					break
				}
				dOff := (dy*surfW + dx) * 4
				if dOff+4 > len(out) {
					break
				}
				out[dOff] = vBarEntry.pixels[srcOff]
				out[dOff+1] = vBarEntry.pixels[srcOff+1]
				out[dOff+2] = vBarEntry.pixels[srcOff+2]
				out[dOff+3] = 0xFF
			}
		}
	}
	return true
}

// decompressSubcodecsData decodes the SUBCODEC layer (a sequence of
// rectangles each compressed with one of subcodec 0/1/2).
//
// FreeRDP: clear_decompress_subcodecs_data (clear.c line 455).
func (ctx *clearCodecCtx) decompressSubcodecsData(r *bytes.Reader, byteCount uint32, surfW, surfH int, out []byte) bool {
	if uint32(r.Len()) < byteCount {
		return false
	}

	var suboffset uint32
	for suboffset < byteCount {
		if r.Len() < 13 {
			return false
		}
		// FreeRDP: clear.c lines 473-478 — note field order matches bands.
		xStart, _ := core.ReadUint16LE(r)
		yStart, _ := core.ReadUint16LE(r)
		width, _ := core.ReadUint16LE(r)
		height, _ := core.ReadUint16LE(r)
		bitmapDataByteCount, _ := core.ReadUInt32LE(r)
		subcodecId, _ := core.ReadUInt8(r)
		suboffset += 13

		if uint32(r.Len()) < bitmapDataByteCount {
			return false
		}

		// Bounds checks (FreeRDP: clear.c lines 486-514).
		if int(xStart)+int(width) > surfW || int(yStart)+int(height) > surfH {
			slog.Debug("ClearCodec: subcodec rect out of bounds",
				"xStart", xStart, "yStart", yStart, "w", width, "h", height)
			return false
		}

		// Read the subcodec payload up-front so the stream advances even
		// if we end up stubbing the decode (e.g. subcodec 1 / NSCodec).
		// This is critical so subsequent rectangles in the same payload
		// stay aligned.
		bmpData, _ := core.ReadBytes(int(bitmapDataByteCount), r)
		suboffset += bitmapDataByteCount

		switch subcodecId {
		case 0:
			// FreeRDP: clear.c lines 521-540 — uncompressed BGR24.
			expected := int(width) * int(height) * 3
			if int(bitmapDataByteCount) != expected {
				slog.Debug("ClearCodec: subcodec 0 byte count mismatch",
					"got", bitmapDataByteCount, "want", expected)
				return false
			}
			rowSrc := int(width) * 3
			for y := 0; y < int(height); y++ {
				srcStart := y * rowSrc
				if srcStart+rowSrc > len(bmpData) {
					break
				}
				dy := int(yStart) + y
				dstStart := (dy*surfW + int(xStart)) * 4
				for x := 0; x < int(width); x++ {
					si := srcStart + x*3
					di := dstStart + x*4
					if di+4 > len(out) {
						return true
					}
					out[di] = bmpData[si]
					out[di+1] = bmpData[si+1]
					out[di+2] = bmpData[si+2]
					out[di+3] = 0xFF
				}
			}

		case 1:
			// FreeRDP: clear.c line 543-549 — CLEARCODEC_SUBCODEC_NSCODEC.
			// The rectangle's bmpData is a self-contained NSCodec
			// payload (header + RLE-encoded planes).  Decode straight
			// into the surface buffer at (xStart, yStart).
			dstOff := (int(yStart)*surfW + int(xStart)) * 4
			if dstOff < 0 || dstOff >= len(out) {
				slog.Debug("ClearCodec subcodec 1 dst offset out of range",
					"off", dstOff, "len", len(out))
				return false
			}
			if !ctx.nsc.Decode(bmpData, int(width), int(height),
				out, surfW*4, int(xStart), int(yStart)) {
				slog.Debug("ClearCodec subcodec 1 (NSCodec) decode failed",
					"rect", []uint16{xStart, yStart, width, height},
					"bytes", bitmapDataByteCount)
				return false
			}

		case 2:
			// FreeRDP: clear.c line 551-556 — CLEARCODEC_SUBCODEC_RLEX.
			if !decompressSubcodeRLEX(bmpData, int(width), int(height),
				out, surfW, int(xStart), int(yStart)) {
				return false
			}

		default:
			slog.Debug("ClearCodec: unknown subcodec id", "id", subcodecId)
			return false
		}
	}
	return true
}

// clear8BitMasks mirrors CLEAR_8BIT_MASKS from clear.c line 83.
var clear8BitMasks = [9]byte{0x00, 0x01, 0x03, 0x07, 0x0F, 0x1F, 0x3F, 0x7F, 0xFF}

// clearLog2Floor mirrors CLEAR_LOG2_FLOOR from clear.c lines 72-81.
// Not stored as a table because it's only consulted once per RLEX call
// for paletteCount-1 ∈ [0,127]; a tiny computed floor(log2) is cheaper.
func clearLog2Floor(v uint32) uint32 {
	r := uint32(0)
	for v > 1 {
		v >>= 1
		r++
	}
	return r
}

// decompressSubcodeRLEX implements CLEARCODEC_SUBCODEC_RLEX.
// FreeRDP: clear_decompress_subcode_rlex (clear.c line 151).
//
// data is the per-rectangle payload (bitmapDataByteCount bytes).  Pixels
// are written into `out` (BGRA32, surfW pixels per row) at offset
// (xDst, yDst) for a (width × height) area.
func decompressSubcodeRLEX(data []byte, width, height int, out []byte, surfW, xDst, yDst int) bool {
	if len(data) < 1 {
		return false
	}
	paletteCount := int(data[0])
	off := 1 + paletteCount*3

	// FreeRDP: clear.c lines 178-182.
	if paletteCount < 1 || paletteCount > 127 {
		slog.Debug("ClearCodec RLEX: invalid paletteCount", "n", paletteCount)
		return false
	}
	if len(data) < off {
		return false
	}

	// Build palette as packed BGRA bytes (4 per entry).
	var palette [128][4]byte
	for i := 0; i < paletteCount; i++ {
		// FreeRDP: clear.c lines 191-195 — B, G, R order in the stream.
		b := data[1+i*3]
		g := data[1+i*3+1]
		r := data[1+i*3+2]
		palette[i] = [4]byte{b, g, r, 0xFF}
	}

	pixelCount := width * height
	var pixelIndex int

	numBits := clearLog2Floor(uint32(paletteCount-1)) + 1
	if numBits > 8 {
		numBits = 8
	}

	x := 0
	y := 0

	writePixel := func(color [4]byte) {
		if x < width && y < height {
			dx := xDst + x
			dy := yDst + y
			di := (dy*surfW + dx) * 4
			if di+4 <= len(out) {
				out[di] = color[0]
				out[di+1] = color[1]
				out[di+2] = color[2]
				out[di+3] = 0xFF
			}
		}
		x++
		if x >= width {
			y++
			x = 0
		}
	}

	for off < len(data) {
		if off+2 > len(data) {
			return false
		}
		tmp := data[off]
		runLengthFactor := uint32(data[off+1])
		off += 2

		// FreeRDP: clear.c lines 214-216.
		suiteDepth := (tmp >> numBits) & clear8BitMasks[8-numBits]
		stopIndex := tmp & clear8BitMasks[numBits]
		startIndex := stopIndex - suiteDepth

		// FreeRDP: clear.c lines 218-234 — extended run length.
		if runLengthFactor >= 0xFF {
			if off+2 > len(data) {
				return false
			}
			runLengthFactor = uint32(data[off]) | uint32(data[off+1])<<8
			off += 2
			if runLengthFactor >= 0xFFFF {
				if off+4 > len(data) {
					return false
				}
				runLengthFactor = uint32(data[off]) |
					uint32(data[off+1])<<8 |
					uint32(data[off+2])<<16 |
					uint32(data[off+3])<<24
				off += 4
			}
		}

		if int(startIndex) >= paletteCount || int(stopIndex) >= paletteCount {
			slog.Debug("ClearCodec RLEX: startIndex/stopIndex out of range",
				"start", startIndex, "stop", stopIndex, "n", paletteCount)
			return false
		}

		suiteIndex := startIndex
		if suiteIndex > 127 {
			return false
		}

		if pixelIndex+int(runLengthFactor) > pixelCount {
			slog.Debug("ClearCodec RLEX: run overflow",
				"pixelIndex", pixelIndex,
				"runLengthFactor", runLengthFactor,
				"pixelCount", pixelCount)
			return false
		}

		// FreeRDP: clear.c lines 269-282 — run of `color` (palette[suiteIndex]).
		color := palette[suiteIndex]
		for i := uint32(0); i < runLengthFactor; i++ {
			writePixel(color)
		}
		pixelIndex += int(runLengthFactor)

		if pixelIndex+int(suiteDepth)+1 > pixelCount {
			slog.Debug("ClearCodec RLEX: suite overflow",
				"pixelIndex", pixelIndex,
				"suiteDepth", suiteDepth,
				"pixelCount", pixelCount)
			return false
		}

		// FreeRDP: clear.c lines 294-318 — suite of palette[startIndex..stopIndex].
		for i := uint32(0); i <= uint32(suiteDepth); i++ {
			if suiteIndex > 127 {
				return false
			}
			c := palette[suiteIndex]
			suiteIndex++
			writePixel(c)
		}
		pixelIndex += int(suiteDepth) + 1
	}

	if pixelIndex != pixelCount {
		slog.Debug("ClearCodec RLEX: pixelIndex != pixelCount",
			"pixelIndex", pixelIndex, "pixelCount", pixelCount)
		return false
	}
	return true
}
