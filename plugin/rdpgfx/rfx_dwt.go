package rdpgfx

// 3-level inverse 2D Discrete Wavelet Transform for the RFX codec.
// Faithful Go port of FreeRDP's libfreerdp/codec/rfx_dwt.c (canonical C
// reference at /tmp/freerdp-ref/rfx_dwt.c in this work tree).
//
// The forward DWT goes vertical-then-horizontal; the inverse goes
// horizontal-then-vertical, in three nested levels (8x8 → 16x16 → 32x32 →
// 64x64).  Each level operates on a contiguous buffer holding the four
// subbands in HL(0), LH(1), HH(2), LL(3) order:
//
//   [HL | LH | HH | LL]   each subband_width × subband_width
//
// The output of one level overwrites the [HL|LH|HH|LL] region and is then
// treated as the new LL for the next level up.  FreeRDP: rfx_dwt.c
// rfx_dwt_2d_decode at line 122.

// rfxInverseDWT2D performs the full 3-level inverse 2D DWT in-place on a
// 4096-element coefficient buffer.  Buffer layout matches FreeRDP's
// rfx_quantization.c comment (and our rfx_quantization.go header).
func rfxInverseDWT2D(coeffs []int16) {
	bufs := idwtBufPool.Get().(*idwtBufs)
	// Level 3: 8×8 subbands → 16×16 output (needs 16×16 = 256 tmp slots).
	// FreeRDP: rfx_dwt.c line 127.
	rfxIDWT2DLevel(coeffs[3840:], bufs.tmp[:256], 8)
	// Level 2: 16×16 → 32×32  (1024 tmp slots).  FreeRDP: line 128.
	rfxIDWT2DLevel(coeffs[3072:], bufs.tmp[:1024], 16)
	// Level 1: 32×32 → 64×64  (4096 tmp slots).  FreeRDP: line 129.
	rfxIDWT2DLevel(coeffs[0:], bufs.tmp[:4096], 32)
	idwtBufPool.Put(bufs)
}

// rfxIDWT2DLevel performs one level of inverse 2D DWT for a subband_width of
// n (one of 8, 16, 32).  buf points at the start of the [HL|LH|HH|LL] layout
// (4n² int16); on return buf holds the (2n)×(2n) reconstructed values.  tmp
// is scratch of length (2n)² supplied by the caller.
//
// This is a literal port of FreeRDP's rfx_dwt_2d_decode_block at rfx_dwt.c
// line 31; the variable names match.  See that file for the lifting-scheme
// derivation.
func rfxIDWT2DLevel(buf, tmp []int16, n int) {
	nn := n * n
	totalWidth := n << 1 // = 2n; the destination row stride

	hl := buf[0:nn]
	lh := buf[nn : 2*nn]
	hh := buf[2*nn : 3*nn]
	ll := buf[3*nn : 4*nn]

	// ─── Horizontal IDWT: read LL/HL/LH/HH columns, write tmp[L|.|H|.] ───
	// FreeRDP: rfx_dwt.c line 50.  L and H half-bands are stored as two halves
	// of tmp, separated by n*totalWidth elements (n rows of width 2n).
	for row := 0; row < n; row++ {
		rowOff := row * n
		lDstOff := row * totalWidth          // L half-band row offset in tmp
		hDstOff := (row + n) * totalWidth    // H half-band row offset in tmp

		// Even coefficients at index 0.  FreeRDP: rfx_dwt.c lines 53-54.
		tmp[lDstOff] = int16(int32(ll[rowOff]) - ((int32(hl[rowOff])+int32(hl[rowOff])+1)>>1))
		tmp[hDstOff] = int16(int32(lh[rowOff]) - ((int32(hh[rowOff])+int32(hh[rowOff])+1)>>1))

		// Even coefficients at indices 2..2n-2.  Lines 55-60.
		for col := 1; col < n; col++ {
			x := col << 1
			tmp[lDstOff+x] = int16(int32(ll[rowOff+col]) - ((int32(hl[rowOff+col-1])+int32(hl[rowOff+col])+1)>>1))
			tmp[hDstOff+x] = int16(int32(lh[rowOff+col]) - ((int32(hh[rowOff+col-1])+int32(hh[rowOff+col])+1)>>1))
		}

		// Odd coefficients at indices 1..2n-3.  Lines 63-73.
		for col := 0; col < n-1; col++ {
			x := col << 1
			ld := (int32(hl[rowOff+col]) << 1) + ((int32(tmp[lDstOff+x]) + int32(tmp[lDstOff+x+2])) >> 1)
			hd := (int32(hh[rowOff+col]) << 1) + ((int32(tmp[hDstOff+x]) + int32(tmp[hDstOff+x+2])) >> 1)
			tmp[lDstOff+x+1] = int16(ld)
			tmp[hDstOff+x+1] = int16(hd)
		}

		// Last odd coefficient at index 2n-1 (boundary: no x+2 sample).
		// FreeRDP: rfx_dwt.c lines 75-80.
		x := (n - 1) << 1
		tmp[lDstOff+x+1] = int16((int32(hl[rowOff+n-1]) << 1) + int32(tmp[lDstOff+x]))
		tmp[hDstOff+x+1] = int16((int32(hh[rowOff+n-1]) << 1) + int32(tmp[hDstOff+x]))
	}

	// ─── Vertical IDWT: read tmp L/H columns, write buf row-major ───
	// FreeRDP: rfx_dwt.c line 92.  L band is the first n rows of tmp; H band
	// is the next n rows.  Output goes back into buf as 2n contiguous rows of
	// length 2n.
	//
	// We process the columns 8 at a time to keep working-set hot in L1; the
	// scalar tail loop handles 2n < 8 (never reached: valid n ∈ {8,16,32} so
	// 2n ∈ {16,32,64}, all multiples of 8).
	const blk = 8
	col := 0
	for ; col+blk <= totalWidth; col += blk {
		// row 0 — boundary at top, no previous H sample.
		// FreeRDP: rfx_dwt.c lines 98-99.
		for b := 0; b < blk; b++ {
			c := col + b
			lVal := int32(tmp[c])
			hVal := int32(tmp[n*totalWidth+c])
			buf[c] = int16(lVal - ((hVal*2 + 1) >> 1))
		}
		// rows 1..n-1 — interior, two coefficients per source row.
		// FreeRDP: rfx_dwt.c lines 101-115.
		for row := 1; row < n; row++ {
			for b := 0; b < blk; b++ {
				c := col + b
				lIdx := row*totalWidth + c
				hIdx := (row+n)*totalWidth + c
				hPrevIdx := (row - 1 + n) * totalWidth + c

				even := int32(tmp[lIdx]) - ((int32(tmp[hPrevIdx]) + int32(tmp[hIdx]) + 1) >> 1)
				buf[2*row*totalWidth+c] = int16(even)

				prevEven := int32(buf[(2*row-2)*totalWidth+c])
				odd := (int32(tmp[hPrevIdx]) << 1) + ((prevEven + even) >> 1)
				buf[(2*row-1)*totalWidth+c] = int16(odd)
			}
		}
		// Last odd at the bottom boundary.  FreeRDP: rfx_dwt.c line 117.
		for b := 0; b < blk; b++ {
			c := col + b
			lastEven := int32(buf[(2*n-2)*totalWidth+c])
			lastH := int32(tmp[(2*n-1)*totalWidth+c])
			buf[(2*n-1)*totalWidth+c] = int16((lastH << 1) + lastEven)
		}
	}
	for ; col < totalWidth; col++ {
		lVal := int32(tmp[col])
		hVal := int32(tmp[n*totalWidth+col])
		buf[col] = int16(lVal - ((hVal*2 + 1) >> 1))
		for row := 1; row < n; row++ {
			lIdx := row*totalWidth + col
			hIdx := (row+n)*totalWidth + col
			hPrevIdx := (row - 1 + n) * totalWidth + col

			even := int32(tmp[lIdx]) - ((int32(tmp[hPrevIdx]) + int32(tmp[hIdx]) + 1) >> 1)
			buf[2*row*totalWidth+col] = int16(even)

			prevEven := int32(buf[(2*row-2)*totalWidth+col])
			odd := (int32(tmp[hPrevIdx]) << 1) + ((prevEven + even) >> 1)
			buf[(2*row-1)*totalWidth+col] = int16(odd)
		}
		lastEven := int32(buf[(2*n-2)*totalWidth+col])
		lastH := int32(tmp[(2*n-1)*totalWidth+col])
		buf[(2*n-1)*totalWidth+col] = int16((lastH << 1) + lastEven)
	}
}

// rfxForwardDWT2D performs the 3-level forward 2D DWT used only by unit
// tests for round-trip verification.  FreeRDP: rfx_dwt.c rfx_dwt_2d_encode at
// line 222.  Order is the reverse of the inverse: 32 → 16 → 8.
func rfxForwardDWT2D(coeffs []int16) {
	bufs := idwtBufPool.Get().(*idwtBufs)
	rfxFDWT2DLevel(coeffs[0:], bufs.tmp[:4096], 32)
	rfxFDWT2DLevel(coeffs[3072:], bufs.tmp[:1024], 16)
	rfxFDWT2DLevel(coeffs[3840:], bufs.tmp[:256], 8)
	idwtBufPool.Put(bufs)
}

// rfxFDWT2DLevel: one level of forward 2D DWT.  Used only by tests; mirrors
// FreeRDP's rfx_dwt_2d_encode_block at rfx_dwt.c line 132.
func rfxFDWT2DLevel(buf, dwt []int16, subbandWidth int) {
	totalWidth := subbandWidth << 1

	// Vertical DWT.  FreeRDP: rfx_dwt.c lines 147-167.
	for x := 0; x < totalWidth; x++ {
		for nn := 0; nn < subbandWidth; nn++ {
			y := nn << 1
			lIdx := nn*totalWidth + x
			hIdx := lIdx + subbandWidth*totalWidth
			srcIdx := y*totalWidth + x

			// H — wrap at boundary uses src[0] instead of src[2*total_width].
			var rightSrc int32
			if nn < subbandWidth-1 {
				rightSrc = int32(buf[srcIdx+2*totalWidth])
			} else {
				rightSrc = int32(buf[srcIdx])
			}
			h := (int32(buf[srcIdx+totalWidth]) - ((int32(buf[srcIdx]) + rightSrc) >> 1)) >> 1
			dwt[hIdx] = int16(h)

			// L — uses dwt[hIdx] and previous-row's H (if any).
			var lowAdj int32
			if nn == 0 {
				lowAdj = h
			} else {
				lowAdj = (int32(dwt[hIdx-totalWidth]) + h) >> 1
			}
			dwt[lIdx] = int16(int32(buf[srcIdx]) + lowAdj)
		}
	}

	// Horizontal DWT.  FreeRDP: rfx_dwt.c lines 169-219.
	for y := 0; y < subbandWidth; y++ {
		rowOff := y * subbandWidth
		// L half-band → HL and LL outputs.
		for nn := 0; nn < subbandWidth; nn++ {
			x := nn << 1
			lSrc := dwt[y*totalWidth+x]
			var rightL int32
			if nn < subbandWidth-1 {
				rightL = int32(dwt[y*totalWidth+x+2])
			} else {
				rightL = int32(dwt[y*totalWidth+x])
			}
			hl := (int32(dwt[y*totalWidth+x+1]) - ((int32(lSrc) + rightL) >> 1)) >> 1
			buf[rowOff+nn] = int16(hl)
			var llAdj int32
			if nn == 0 {
				llAdj = hl
			} else {
				llAdj = (int32(buf[rowOff+nn-1]) + hl) >> 1
			}
			buf[3*subbandWidth*subbandWidth+rowOff+nn] = int16(int32(lSrc) + llAdj)
		}
		// H half-band → HH and LH outputs.
		for nn := 0; nn < subbandWidth; nn++ {
			x := nn << 1
			hSrc := dwt[(y+subbandWidth)*totalWidth+x]
			var rightH int32
			if nn < subbandWidth-1 {
				rightH = int32(dwt[(y+subbandWidth)*totalWidth+x+2])
			} else {
				rightH = int32(dwt[(y+subbandWidth)*totalWidth+x])
			}
			hh := (int32(dwt[(y+subbandWidth)*totalWidth+x+1]) - ((int32(hSrc) + rightH) >> 1)) >> 1
			buf[2*subbandWidth*subbandWidth+rowOff+nn] = int16(hh)
			var lhAdj int32
			if nn == 0 {
				lhAdj = hh
			} else {
				lhAdj = (int32(buf[subbandWidth*subbandWidth+rowOff+nn-1]) + hh) >> 1
			}
			buf[subbandWidth*subbandWidth+rowOff+nn] = int16(int32(hSrc) + lhAdj)
		}
	}
}
