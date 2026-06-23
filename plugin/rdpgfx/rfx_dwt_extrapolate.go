package rdpgfx

// 3-level inverse 2D DWT for the RFX Progressive codec's
// RFX_DWT_REDUCE_EXTRAPOLATE band layout.  Faithful Go port of FreeRDP's
// libfreerdp/codec/progressive.c helpers:
//
//   rfx_dwt_2d_extrapolate_decode          (progressive.c line 800)
//   progressive_rfx_dwt_2d_decode_block    (progressive.c line 757)
//   progressive_rfx_idwt_x                 (progressive.c line 600)
//   progressive_rfx_idwt_y                 (progressive.c line 669)
//   progressive_rfx_get_band_l_count       (progressive.c line 744)
//   progressive_rfx_get_band_h_count       (progressive.c line 749)
//
// Unlike the non-extrapolate inverse DWT in rfx_dwt.go (which uses the fixed
// 1024/256/64 band layout), the extrapolate variant uses asymmetric band
// counts per level so that the reconstructed low-pass band is one sample wider
// in each dimension and the high bands one sample narrower:
//
//   level 1: nBandL=33 nBandH=31  HL=1023 LH=1023 HH=961  LL=1089 (33x33)
//   level 2: nBandL=17 nBandH=16  HL=272  LH=272  HH=256  LL=289  (17x17)
//   level 3: nBandL=9  nBandH=8   HL=72   LH=72   HH=64   LL=81   (9x9)
//
// Buffer offsets within the 4096-coeff tile (from the decode_component
// extrapolate branch):
//
//   HL1 0,    LH1 1023, HH1 2046,
//   HL2 3007, LH2 3279, HH2 3551,
//   HL3 3807, LH3 3879, HH3 3951,
//   LL3 4015
//
// The decode runs level 3 first (&buffer[3807]), then level 2 (&buffer[3007]),
// then level 1 (&buffer[0]); each level reconstructs its (nBandL+nBandH)^2 LL
// output in place at the band base, which becomes the LL input of the next
// (lower) level.

// progressiveBandLCount returns (64>>level)+1.
// FreeRDP: progressive.c progressive_rfx_get_band_l_count at line 744.
func progressiveBandLCount(level int) int {
	return (64 >> uint(level)) + 1
}

// progressiveBandHCount.
// FreeRDP: progressive.c progressive_rfx_get_band_h_count at line 749.
func progressiveBandHCount(level int) int {
	if level == 1 {
		return (64 >> 1) - 1
	}
	return (64 + (1 << uint(level-1))) >> uint(level)
}

// rfxInverseDWT2DExtrapolate performs the full 3-level extrapolate inverse 2D
// DWT in-place on a 4096-element coefficient buffer.  FreeRDP:
// rfx_dwt_2d_extrapolate_decode at progressive.c line 800.
func rfxInverseDWT2DExtrapolate(coeffs []int16) {
	bufs := idwtBufPool.Get().(*idwtBufs)
	tmp := bufs.tmp[:4096]
	progressiveDWT2DDecodeBlock(coeffs[3807:], tmp, 3)
	progressiveDWT2DDecodeBlock(coeffs[3007:], tmp, 2)
	progressiveDWT2DDecodeBlock(coeffs[0:], tmp, 1)
	idwtBufPool.Put(bufs)
}

// progressiveDWT2DDecodeBlock decodes one level of the extrapolate inverse 2D
// DWT.  buffer points at the band base (HL of this level); temp is scratch of
// at least (nBandL+nBandH)^2 int16.  FreeRDP: progressive.c
// progressive_rfx_dwt_2d_decode_block at line 757.
func progressiveDWT2DDecodeBlock(buffer, temp []int16, level int) {
	nBandL := progressiveBandLCount(level)
	nBandH := progressiveBandHCount(level)

	offset := 0
	hl := buffer[offset:]
	offset += nBandH * nBandL
	lh := buffer[offset:]
	offset += nBandL * nBandH
	hh := buffer[offset:]
	offset += nBandH * nBandH
	ll := buffer[offset:]

	nDstStepX := nBandL + nBandH
	nDstStepY := nBandL + nBandH

	l := temp[0:]
	h := temp[nBandL*nDstStepX:]
	llx := buffer[0:]

	// horizontal (LL + HL -> L)
	progressiveIDWTx(ll, nBandL, hl, nBandH, l, nDstStepX, nBandL, nBandH, nBandL)
	// horizontal (LH + HH -> H)
	progressiveIDWTx(lh, nBandL, hh, nBandH, h, nDstStepX, nBandL, nBandH, nBandH)
	// vertical (L + H -> LL)
	progressiveIDWTy(l, nDstStepX, h, nDstStepX, llx, nDstStepY, nBandL, nBandH, nBandL+nBandH)
}

// progressiveIDWTx is the horizontal inverse DWT lifting step.  Each of
// nDstCount source rows produces a destination row written with stride 2 (the
// even/odd interleave) into pDstBand, advancing pLowBand/pHighBand/pDstBand by
// their respective steps after each row.  FreeRDP: progressive.c
// progressive_rfx_idwt_x at line 600.
func progressiveIDWTx(pLowBand []int16, nLowStep int, pHighBand []int16, nHighStep int,
	pDstBand []int16, nDstStep int, nLowCount, nHighCount, nDstCount int) {

	lowBase := 0
	highBase := 0
	dstBase := 0

	for i := 0; i < nDstCount; i++ {
		pl := lowBase
		ph := highBase
		px := dstBase

		h0 := int32(pHighBand[ph])
		ph++
		l0 := int32(pLowBand[pl])
		pl++
		x0 := int32(clampInt16(l0 - h0))
		x2 := int32(clampInt16(l0 - h0))

		for j := 0; j < nHighCount-1; j++ {
			h1 := int32(pHighBand[ph])
			ph++
			l0 = int32(pLowBand[pl])
			pl++
			x2 = int32(clampInt16(l0 - ((h0 + h1) / 2)))
			x1 := int32(clampInt16((x0+x2)/2 + (2 * h0)))
			pDstBand[px+0] = int16(x0)
			pDstBand[px+1] = int16(x1)
			px += 2
			x0 = x2
			h0 = h1
		}

		if nLowCount <= nHighCount+1 {
			if nLowCount <= nHighCount {
				pDstBand[px+0] = int16(x2)
				pDstBand[px+1] = clampInt16(x2 + (2 * h0))
			} else {
				l0 = int32(pLowBand[pl])
				pl++
				x0 = int32(clampInt16(l0 - h0))
				pDstBand[px+0] = int16(x2)
				pDstBand[px+1] = clampInt16((x0+x2)/2 + (2 * h0))
				pDstBand[px+2] = int16(x0)
			}
		} else {
			l0 = int32(pLowBand[pl])
			pl++
			x0 = int32(clampInt16(l0 - (h0 / 2)))
			pDstBand[px+0] = int16(x2)
			pDstBand[px+1] = clampInt16((x0+x2)/2 + (2 * h0))
			pDstBand[px+2] = int16(x0)
			l0 = int32(pLowBand[pl])
			pl++
			pDstBand[px+3] = clampInt16((x0 + l0) / 2)
		}

		lowBase += nLowStep
		highBase += nHighStep
		dstBase += nDstStep
	}
}

// progressiveIDWTy is the vertical inverse DWT lifting step.  It walks
// nDstCount columns; within each column it steps through the low/high source
// bands and destination by their respective strides (columns are non-unit
// stride here, rows are unit stride, mirroring the C pointer arithmetic).
// FreeRDP: progressive.c progressive_rfx_idwt_y at line 669.
func progressiveIDWTy(pLowBand []int16, nLowStep int, pHighBand []int16, nHighStep int,
	pDstBand []int16, nDstStep int, nLowCount, nHighCount, nDstCount int) {

	lowBase := 0
	highBase := 0
	dstBase := 0

	for i := 0; i < nDstCount; i++ {
		pl := lowBase
		ph := highBase
		px := dstBase

		h0 := int32(pHighBand[ph])
		ph += nHighStep
		l0 := int32(pLowBand[pl])
		pl += nLowStep
		x0 := int32(clampInt16(l0 - h0))
		x2 := int32(clampInt16(l0 - h0))

		for j := 0; j < nHighCount-1; j++ {
			h1 := int32(pHighBand[ph])
			ph += nHighStep
			l0 = int32(pLowBand[pl])
			pl += nLowStep
			x2 = int32(clampInt16(l0 - ((h0 + h1) / 2)))
			x1 := int32(clampInt16((x0+x2)/2 + (2 * h0)))
			pDstBand[px] = int16(x0)
			px += nDstStep
			pDstBand[px] = int16(x1)
			px += nDstStep
			x0 = x2
			h0 = h1
		}

		if nLowCount <= nHighCount+1 {
			if nLowCount <= nHighCount {
				pDstBand[px] = int16(x2)
				px += nDstStep
				pDstBand[px] = clampInt16(x2 + (2 * h0))
			} else {
				l0 = int32(pLowBand[pl])
				x0 = int32(clampInt16(l0 - h0))
				pDstBand[px] = int16(x2)
				px += nDstStep
				pDstBand[px] = clampInt16((x0+x2)/2 + (2 * h0))
				px += nDstStep
				pDstBand[px] = int16(x0)
			}
		} else {
			l0 = int32(pLowBand[pl])
			pl += nLowStep
			x0 = int32(clampInt16(l0 - (h0 / 2)))
			pDstBand[px] = int16(x2)
			px += nDstStep
			pDstBand[px] = clampInt16((x0+x2)/2 + (2 * h0))
			px += nDstStep
			pDstBand[px] = int16(x0)
			px += nDstStep
			l0 = int32(pLowBand[pl])
			pDstBand[px] = clampInt16((x0 + l0) / 2)
		}

		lowBase++
		highBase++
		dstBase++
	}
}
