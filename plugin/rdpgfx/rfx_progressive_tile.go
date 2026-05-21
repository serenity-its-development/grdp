package rdpgfx

// Per-tile decode for the RFX progressive codec: TILE_SIMPLE, TILE_FIRST, and
// TILE_UPGRADE.  Faithful Go port of the corresponding helpers in FreeRDP's
// libfreerdp/codec/progressive.c (canonical C at /tmp/freerdp-ref/progressive.c
// in this work tree); each function carries a `// FreeRDP: progressive.c line N`
// pointer back at the C function it mirrors.
//
// Bands and offsets match rfx_quantization.go's header:
//
//   HL1 0..1024, LH1 1024..2048, HH1 2048..3072,
//   HL2 3072..3328, LH2 3328..3584, HH2 3584..3840,
//   HL3 3840..3904, LH3 3904..3968, HH3 3968..4032,
//   LL3 4032..4096

import (
	"log/slog"
	"sync"
)

// ─── Tile parsing & dispatch ──────────────────────────────────────────────

// decodeTileSimple handles PROGRESSIVE_WBT_TILE_SIMPLE (0xCCC5).  Wire format
// (16-byte fixed header + Y/Cb/Cr/tail blocks):
//
//	quantIdxY(1) quantIdxCb(1) quantIdxCr(1)
//	xIdx(2) yIdx(2) flags(1)
//	yLen(2) cbLen(2) crLen(2) tailLen(2)
//	yData(yLen) cbData(cbLen) crData(crLen) tailData(tailLen)
//
// FreeRDP: progressive.c progressive_tile_read (simple=TRUE) at line 1593, and
// progressive_decompress_tile_first (which handles both FIRST and SIMPLE) at
// line 929.
func (d *rfxProgressiveDecoder) decodeTileSimple(data []byte, quants []rfxQuant, surface *progressiveSurface, output []byte, outW, outH int, parallelComponents bool) {
	if len(data) < 16 {
		return
	}
	quantIdxY := int(data[0])
	quantIdxCb := int(data[1])
	quantIdxCr := int(data[2])
	xIdx := leU16(data[3:])
	yIdx := leU16(data[5:])
	flags := data[7]
	yLen := int(leU16(data[8:]))
	cbLen := int(leU16(data[10:]))
	crLen := int(leU16(data[12:]))
	// tailLen at data[14:16] — ignored, matching FreeRDP.

	off := 16
	yData := safeSlice(data, off, yLen)
	off += yLen
	cbData := safeSlice(data, off, cbLen)
	off += cbLen
	crData := safeSlice(data, off, crLen)

	d.decodeFirstPass(surface, int(xIdx), int(yIdx), flags, 0xFF,
		quants, nil, quantIdxY, quantIdxCb, quantIdxCr,
		yData, cbData, crData, output, outW, outH, parallelComponents)
}

// decodeTileFirst handles PROGRESSIVE_WBT_TILE_FIRST (0xCCC6).  Wire format
// (17-byte fixed header):
//
//	quantIdxY(1) quantIdxCb(1) quantIdxCr(1)
//	xIdx(2) yIdx(2) flags(1) quality(1)
//	yLen(2) cbLen(2) crLen(2) tailLen(2)
//	yData(yLen) cbData(cbLen) crData(crLen) tailData(tailLen)
//
// FreeRDP: progressive.c progressive_tile_read (simple=FALSE) at line 1593.
func (d *rfxProgressiveDecoder) decodeTileFirst(data []byte, quants []rfxQuant, progQuants []rfxProgQuant, surface *progressiveSurface, output []byte, outW, outH int, parallelComponents bool) {
	if len(data) < 17 {
		return
	}
	quantIdxY := int(data[0])
	quantIdxCb := int(data[1])
	quantIdxCr := int(data[2])
	xIdx := leU16(data[3:])
	yIdx := leU16(data[5:])
	flags := data[7]
	quality := data[8]
	yLen := int(leU16(data[9:]))
	cbLen := int(leU16(data[11:]))
	crLen := int(leU16(data[13:]))
	// tailLen at data[15:17] — ignored.

	off := 17
	yData := safeSlice(data, off, yLen)
	off += yLen
	cbData := safeSlice(data, off, cbLen)
	off += cbLen
	crData := safeSlice(data, off, crLen)

	d.decodeFirstPass(surface, int(xIdx), int(yIdx), flags, quality,
		quants, progQuants, quantIdxY, quantIdxCb, quantIdxCr,
		yData, cbData, crData, output, outW, outH, parallelComponents)
}

// decodeFirstPass is the shared FIRST/SIMPLE backend (FreeRDP merges them via
// progressive_decompress_tile_first at progressive.c line 929).  It populates
// the per-tile cache with post-shift coefficient state (so future UPGRADEs
// have something to refine) and emits the BGRA tile.
//
// When tile flags contain RFX_TILE_DIFFERENCE (coeffDiff=TRUE in FreeRDP),
// the FIRST pass's coefficients are deltas to add on top of the previously
// stored current[]; otherwise current[] is overwritten.  FreeRDP:
// progressive.c progressive_rfx_dwt_2d_decode lines 821-830.
func (d *rfxProgressiveDecoder) decodeFirstPass(surface *progressiveSurface,
	xIdx, yIdx int, flags, quality uint8,
	quants []rfxQuant, progQuants []rfxProgQuant,
	quantIdxY, quantIdxCb, quantIdxCr int,
	yData, cbData, crData []byte, output []byte, outW, outH int,
	parallelComponents bool,
) {
	tile := surface.getTile(xIdx, yIdx)
	if tile == nil {
		return
	}
	coeffDiff := (flags & rfxTileDifference) != 0 && tile.allocated

	qY := rfxGetQuant(quants, quantIdxY)
	qCb := rfxGetQuant(quants, quantIdxCb)
	qCr := rfxGetQuant(quants, quantIdxCr)

	// Look up progressive quant triple; quality 0xFF means "full quality"
	// (TILE_SIMPLE always carries 0xFF), encoded with all-zero progressive
	// quant offsets.  FreeRDP: progressive.c lines 997-1011.
	var qpY, qpCb, qpCr rfxQuant
	if quality != 0xFF {
		if int(quality) < len(progQuants) {
			qpY = progQuants[quality].y
			qpCb = progQuants[quality].cb
			qpCr = progQuants[quality].cr
		} else {
			slog.Debug("RFX progressive: quality index out of range", "quality", quality, "have", len(progQuants))
		}
	}

	// Per-component shift = (quant + progQuant) − 1.  FreeRDP: progressive.c
	// lines 1024-1035.
	shiftY := addQuant(qY, qpY)
	shiftCb := addQuant(qCb, qpCb)
	shiftCr := addQuant(qCr, qpCr)
	shiftY, _ = quantLsub(shiftY, 1)
	shiftCb, _ = quantLsub(shiftCb, 1)
	shiftCr, _ = quantLsub(shiftCr, 1)

	// Decode the three components.  Each decode populates tile.current[c]
	// and tile.sign[c], then leaves a transformed 4096-element coefficient
	// slice in the local pooled buffer ready for blitting.
	var yCoeffs, cbCoeffs, crCoeffs []int16
	if parallelComponents {
		var wg sync.WaitGroup
		wg.Go(func() { yCoeffs = decodeComponentFirst(yData, shiftY, tile, 0, coeffDiff) })
		wg.Go(func() { cbCoeffs = decodeComponentFirst(cbData, shiftCb, tile, 1, coeffDiff) })
		wg.Go(func() { crCoeffs = decodeComponentFirst(crData, shiftCr, tile, 2, coeffDiff) })
		wg.Wait()
	} else {
		yCoeffs = decodeComponentFirst(yData, shiftY, tile, 0, coeffDiff)
		cbCoeffs = decodeComponentFirst(cbData, shiftCb, tile, 1, coeffDiff)
		crCoeffs = decodeComponentFirst(crData, shiftCr, tile, 2, coeffDiff)
	}

	// Record cumulative bit-position for future UPGRADE passes.  FreeRDP:
	// progressive.c lines 1017-1026.
	tile.yQuant = qY
	tile.cbQuant = qCb
	tile.crQuant = qCr
	tile.yBitPos = addQuant(qY, qpY)
	tile.cbBitPos = addQuant(qCb, qpCb)
	tile.crBitPos = addQuant(qCr, qpCr)
	tile.pass = 1
	tile.allocated = true

	rfxPlaceTile(yCoeffs, cbCoeffs, crCoeffs, xIdx, yIdx, output, outW, outH)

	coeffPool.Put((*coeffArr)(yCoeffs))
	coeffPool.Put((*coeffArr)(cbCoeffs))
	coeffPool.Put((*coeffArr)(crCoeffs))
}

// decodeTileUpgrade handles PROGRESSIVE_WBT_TILE_UPGRADE (0xCCC7).  Wire
// format (20-byte fixed header + six per-component data segments):
//
//	quantIdxY(1) quantIdxCb(1) quantIdxCr(1)
//	xIdx(2) yIdx(2) quality(1)
//	ySrlLen(2) yRawLen(2) cbSrlLen(2) cbRawLen(2) crSrlLen(2) crRawLen(2)
//	ySrl(ySrlLen) yRaw(yRawLen) cbSrl(...) cbRaw(...) crSrl(...) crRaw(...)
//
// FreeRDP: progressive.c progressive_tile_read_upgrade at line 1515 and
// progressive_decompress_tile_upgrade at line 1331.
func (d *rfxProgressiveDecoder) decodeTileUpgrade(data []byte, quants []rfxQuant, progQuants []rfxProgQuant, surface *progressiveSurface, output []byte, outW, outH int, ctxFlags uint8, parallelComponents bool) {
	if len(data) < 20 {
		return
	}
	quantIdxY := int(data[0])
	quantIdxCb := int(data[1])
	quantIdxCr := int(data[2])
	xIdx := leU16(data[3:])
	yIdx := leU16(data[5:])
	quality := data[7]
	ySrlLen := int(leU16(data[8:]))
	yRawLen := int(leU16(data[10:]))
	cbSrlLen := int(leU16(data[12:]))
	cbRawLen := int(leU16(data[14:]))
	crSrlLen := int(leU16(data[16:]))
	crRawLen := int(leU16(data[18:]))

	off := 20
	ySrl := safeSlice(data, off, ySrlLen)
	off += ySrlLen
	yRaw := safeSlice(data, off, yRawLen)
	off += yRawLen
	cbSrl := safeSlice(data, off, cbSrlLen)
	off += cbSrlLen
	cbRaw := safeSlice(data, off, cbRawLen)
	off += cbRawLen
	crSrl := safeSlice(data, off, crSrlLen)
	off += crSrlLen
	crRaw := safeSlice(data, off, crRawLen)

	tile := surface.getTile(int(xIdx), int(yIdx))
	if tile == nil || !tile.allocated {
		slog.Debug("RFX progressive: UPGRADE for tile without prior FIRST", "xIdx", xIdx, "yIdx", yIdx)
		return
	}

	// FreeRDP warns (but proceeds) when SUBBAND_DIFFING is not set.  We mirror
	// that posture — upgrade only makes sense when the prior pass's coefficients
	// are being differentially refined.
	_ = ctxFlags

	qY := rfxGetQuant(quants, quantIdxY)
	qCb := rfxGetQuant(quants, quantIdxCb)
	qCr := rfxGetQuant(quants, quantIdxCr)

	var qpY, qpCb, qpCr rfxQuant
	if quality != 0xFF {
		if int(quality) < len(progQuants) {
			qpY = progQuants[quality].y
			qpCb = progQuants[quality].cb
			qpCr = progQuants[quality].cr
		}
	}

	// New bit-positions and per-band number-of-extra-bits delta.  FreeRDP:
	// progressive.c lines 1439-1456.
	newYBitPos := addQuant(qY, qpY)
	newCbBitPos := addQuant(qCb, qpCb)
	newCrBitPos := addQuant(qCr, qpCr)
	yNumBits, okY := subQuant(tile.yBitPos, newYBitPos)
	cbNumBits, okCb := subQuant(tile.cbBitPos, newCbBitPos)
	crNumBits, okCr := subQuant(tile.crBitPos, newCrBitPos)
	if !okY || !okCb || !okCr {
		slog.Debug("RFX progressive: UPGRADE bit-position underflow", "xIdx", xIdx, "yIdx", yIdx)
	}

	shiftY, _ := quantLsub(addQuant(qY, qpY), 1)
	shiftCb, _ := quantLsub(addQuant(qCb, qpCb), 1)
	shiftCr, _ := quantLsub(addQuant(qCr, qpCr), 1)

	tile.pass++

	var yOut, cbOut, crOut []int16
	if parallelComponents {
		var wg sync.WaitGroup
		wg.Go(func() { yOut = decodeComponentUpgrade(tile, 0, shiftY, yNumBits, ySrl, yRaw) })
		wg.Go(func() { cbOut = decodeComponentUpgrade(tile, 1, shiftCb, cbNumBits, cbSrl, cbRaw) })
		wg.Go(func() { crOut = decodeComponentUpgrade(tile, 2, shiftCr, crNumBits, crSrl, crRaw) })
		wg.Wait()
	} else {
		yOut = decodeComponentUpgrade(tile, 0, shiftY, yNumBits, ySrl, yRaw)
		cbOut = decodeComponentUpgrade(tile, 1, shiftCb, cbNumBits, cbSrl, cbRaw)
		crOut = decodeComponentUpgrade(tile, 2, shiftCr, crNumBits, crSrl, crRaw)
	}

	tile.yQuant = qY
	tile.cbQuant = qCb
	tile.crQuant = qCr
	tile.yBitPos = newYBitPos
	tile.cbBitPos = newCbBitPos
	tile.crBitPos = newCrBitPos

	rfxPlaceTile(yOut, cbOut, crOut, int(xIdx), int(yIdx), output, outW, outH)

	coeffPool.Put((*coeffArr)(yOut))
	coeffPool.Put((*coeffArr)(cbOut))
	coeffPool.Put((*coeffArr)(crOut))
}

// ─── Component decode (FIRST / SIMPLE) ────────────────────────────────────

// decodeComponentFirst RLGR-decodes one component into a 4096-coeff buffer,
// captures the pre-shift signed magnitudes into tile.sign[c], applies the
// differential decode + per-band quant shift to obtain post-shift current[c],
// then runs the inverse DWT.  Returns a pooled []int16 (caller must Put back).
//
// FreeRDP: progressive.c progressive_rfx_decode_component at line 860.
// When coeffDiff is true, the new post-shift coefficients are *added* to the
// previously cached current[comp] rather than overwriting it (FreeRDP:
// progressive.c lines 824-830).  In that case the sign buffer is kept as-is
// (the sign tracking is only meaningful for the very first FIRST pass on a
// given tile).
func decodeComponentFirst(data []byte, shift rfxQuant, tile *progressiveTile, comp int, coeffDiff bool) []int16 {
	arr := coeffPool.Get().(*coeffArr)
	coeffs := arr[:]

	if data == nil {
		clear(coeffs)
		if !coeffDiff {
			clear(tile.current[comp][:])
			clear(tile.sign[comp][:])
		}
		// Inverse DWT of zeros is zeros — skip when there's nothing to refine.
		if coeffDiff {
			copy(coeffs, tile.current[comp][:])
			rfxInverseDWT2D(coeffs)
		}
		return coeffs
	}

	// 1. RLGR1 entropy decode → 4096 signed coefficients.  Progressive always
	// uses RLGR1 (CLW_ENTROPY_RLGR1) per FreeRDP's hard-coded call at
	// progressive.c line 871.
	coeffs = rlgr1Decode(data, 4096, coeffs)

	// 2. Snapshot the pre-shift values into tile.sign[c] for future UPGRADE
	// passes to read.  FreeRDP: progressive.c line 876 (CopyMemory before
	// any shift/differential transform).
	copy(tile.sign[comp][:], coeffs)

	// 3. Differential decode LL3 (cumulative sum across the 64 LL3 coeffs)
	// fused with the LL3 quant shift via cumsum(x)*2^s == cumsum(x*2^s).
	// FreeRDP: progressive.c line 879 (rfx_differential_decode + later
	// rfx_quantization_decode applies shift->LL3).
	if shift.LL3 > 0 {
		s := uint(shift.LL3)
		coeffs[4032] = int16(int32(coeffs[4032]) << s)
		for i := 4033; i < 4096; i++ {
			coeffs[i] = coeffs[i-1] + int16(int32(coeffs[i])<<s)
		}
	} else {
		for i := 4033; i < 4096; i++ {
			coeffs[i] += coeffs[i-1]
		}
	}

	// 4. Apply per-band quant shift to every band except LL3 (handled above).
	// FreeRDP: progressive.c lines 880-898.
	rfxDequantizeSkipLL3(coeffs, shift)

	// 5. Capture post-shift coefficients for UPGRADE refinement.  FreeRDP:
	// dwt_2d_decode calls memcpy(current, buffer, ...) when coeffDiff=FALSE
	// (progressive.c line 824).  When coeffDiff is TRUE the new coeffs are
	// added on top of the cached current[] (progressive.c lines 827-830).
	if coeffDiff {
		cur := tile.current[comp][:]
		for i := range coeffs {
			summed := int32(coeffs[i]) + int32(cur[i])
			coeffs[i] = clampInt16(summed)
		}
		copy(cur, coeffs)
	} else {
		copy(tile.current[comp][:], coeffs)
	}

	// 6. Inverse 3-level 2D DWT in-place.  FreeRDP: progressive.c line 925.
	rfxInverseDWT2D(coeffs)
	return coeffs
}

// ─── Component decode (UPGRADE) ────────────────────────────────────────────

// decodeComponentUpgrade applies one progressive UPGRADE refinement pass to a
// previously decoded tile component.  shift gives the per-band left-shift
// count to apply to each incoming refinement bit (= the same shift used by
// the FIRST pass's quantization, since UPGRADE bits land in the same
// magnitude positions).  numBits gives, for each band, how many additional
// precision bits the encoder is providing this pass.  srlData is the SRL
// (sign-run-length) bitstream used for non-LL coefficients that were zero in
// the prior pass; rawData is the RAW bitstream used everywhere else.  After
// refinement, current[comp] is copied into the working buffer and IDWT runs.
//
// Note: this function expects the *non-extrapolate* band layout (same as the
// FIRST pass): HL1@0..1024, LH1@1024..2048, HH1@2048..3072, etc.  FreeRDP's
// upgrade_component uses the extrapolate layout (1023/961/272/72/...) because
// progressive *always* sends UPGRADE with RFX_DWT_REDUCE_EXTRAPOLATE in
// modern captures.  We reject extrapolate at the region level and rely on
// the legacy 1024/256/64 layout for both passes.  Real Windows servers using
// progressive without extrapolate still send UPGRADE blocks; this is the
// path we exercise.  FreeRDP: progressive.c progressive_rfx_upgrade_component
// at line 1258.
func decodeComponentUpgrade(tile *progressiveTile, comp int, shift rfxQuant, numBits rfxQuant, srlData, rawData []byte) []int16 {
	arr := coeffPool.Get().(*coeffArr)
	out := arr[:]

	state := newProgressiveUpgradeState(srlData, rawData)

	current := tile.current[comp][:]
	sign := tile.sign[comp][:]

	// Non-LL bands (sign-tracked, may pull from SRL stream).  FreeRDP:
	// progressive.c lines 1282-1316.
	state.nonLL = true
	state.refineBlock(current[0:1024], sign[0:1024], shift.HL1, numBits.HL1)
	state.refineBlock(current[1024:2048], sign[1024:2048], shift.LH1, numBits.LH1)
	state.refineBlock(current[2048:3072], sign[2048:3072], shift.HH1, numBits.HH1)
	state.refineBlock(current[3072:3328], sign[3072:3328], shift.HL2, numBits.HL2)
	state.refineBlock(current[3328:3584], sign[3328:3584], shift.LH2, numBits.LH2)
	state.refineBlock(current[3584:3840], sign[3584:3840], shift.HH2, numBits.HH2)
	state.refineBlock(current[3840:3904], sign[3840:3904], shift.HL3, numBits.HL3)
	state.refineBlock(current[3904:3968], sign[3904:3968], shift.LH3, numBits.LH3)
	state.refineBlock(current[3968:4032], sign[3968:4032], shift.HH3, numBits.HH3)

	// LL3 band — RAW only, no sign tracking.  FreeRDP: progressive.c line 1320.
	state.nonLL = false
	state.refineBlock(current[4032:4096], sign[4032:4096], shift.LL3, numBits.LL3)

	// Copy refined current[c] into the working buffer and run IDWT.
	// FreeRDP: progressive.c line 1327 dwt_2d_decode(buffer, current,
	// coeffDiff, extrapolate, reverse=TRUE) → memcpy(buffer, current, ...).
	copy(out, current)
	rfxInverseDWT2D(out)
	return out
}

// ─── Output blit ──────────────────────────────────────────────────────────

// rfxPlaceTile converts a 64×64 YCbCr tile to BGRA using ICT and writes it to
// the surface at tile-grid coordinates (xIdx, yIdx).  Edge tiles get clipped
// to the surface width/height.
func rfxPlaceTile(yCoeffs, cbCoeffs, crCoeffs []int16, xIdx, yIdx int, output []byte, outW, outH int) {
	rfxPlaceTileAbs(yCoeffs, cbCoeffs, crCoeffs, xIdx*rfxTileSize, yIdx*rfxTileSize, output, outW, outH)
}

// rfxPlaceTileAbs is the absolute-pixel-coordinates variant; used both by the
// progressive tile placement above and by the non-progressive rfx.go.
func rfxPlaceTileAbs(yCoeffs, cbCoeffs, crCoeffs []int16, tileX, tileY int, output []byte, outW, outH int) {
	tileW := rfxTileSize
	tileH := rfxTileSize
	if tileX+tileW > outW {
		tileW = outW - tileX
	}
	if tileY+tileH > outH {
		tileH = outH - tileY
	}
	if tileW <= 0 || tileH <= 0 {
		return
	}

	for row := 0; row < tileH; row++ {
		dstStart := ((tileY+row)*outW + tileX) * 4
		dstEnd := dstStart + tileW*4
		if dstStart < 0 || dstEnd > len(output) {
			continue
		}
		dstRow := output[dstStart:dstEnd:dstEnd]
		srcOff := row * rfxTileSize
		ictToBGRA(
			yCoeffs[srcOff:srcOff+tileW:srcOff+tileW],
			cbCoeffs[srcOff:srcOff+tileW:srcOff+tileW],
			crCoeffs[srcOff:srcOff+tileW:srcOff+tileW],
			dstRow, tileW,
		)
	}
}

// rfxDecodeComponent is the non-progressive RFX entry point retained for
// rfx.go's decodeTile.  Progressive tiles use decodeComponentFirst /
// decodeComponentUpgrade directly.
func rfxDecodeComponent(data []byte, quant rfxQuant, rlgrMode int) []int16 {
	arr := coeffPool.Get().(*coeffArr)
	coeffs := arr[:]

	if data == nil {
		clear(coeffs)
		return coeffs
	}

	if rlgrMode == 3 {
		coeffs = rlgr3Decode(data, 4096, coeffs)
	} else {
		coeffs = rlgr1Decode(data, 4096, coeffs)
	}

	// Differential decode + LL3 quant shift fused (cumsum identity).  Match
	// FreeRDP: rfx_decode.c uses the same logic for non-progressive tiles.
	q := quant
	if q.LL3 > 1 {
		s := uint(q.LL3 - 1)
		coeffs[4032] = int16(int32(coeffs[4032]) << s)
		for i := 4033; i < 4096; i++ {
			coeffs[i] = coeffs[i-1] + int16(int32(coeffs[i])<<s)
		}
	} else {
		for i := 4033; i < 4096; i++ {
			coeffs[i] += coeffs[i-1]
		}
	}

	rfxDequantizeSkipLL3FromWire(coeffs, q)

	rfxInverseDWT2D(coeffs)
	return coeffs
}
