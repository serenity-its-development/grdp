package rdpgfx

// Per-band scalar quantization for the RFX (non-progressive *and* progressive)
// codec.  FreeRDP: libfreerdp/codec/rfx_quantization.c — see /tmp/freerdp-ref
// in this work tree.
//
// Coefficient buffer layout for both variants:
//
//   Band   Offset  Dimensions  Size
//   HL1    0       32x32       1024
//   LH1    1024    32x32       1024
//   HH1    2048    32x32       1024
//   HL2    3072    16x16       256
//   LH2    3328    16x16       256
//   HH2    3584    16x16       256
//   HL3    3840    8x8         64
//   LH3    3904    8x8         64
//   HH3    3968    8x8         64
//   LL3    4032    8x8         64
//
// (FreeRDP: rfx_quantization.c lines 31-45.)
//
// Each per-band shift is an *unsigned* left-shift count; FreeRDP arrives at
// the shift by subtracting 1 from the wire quant value (or, for progressive,
// (quant + progQuant) − 1).  A zero shift is a no-op.  Shifts above 15 would
// overflow int16 and shouldn't occur in valid streams; we clamp defensively.

// rfxShiftSubband left-shifts each int16 in-place by `shift` bits.  shift==0
// is a no-op; shift>15 is clamped.  FreeRDP: rfx_quantization.c
// rfx_quantization_decode_block at line 47.
func rfxShiftSubband(data []int16, shift uint8) {
	if shift == 0 {
		return
	}
	if shift > 15 {
		shift = 15
	}
	s := uint(shift)
	for i := range data {
		data[i] = int16(int32(data[i]) << s)
	}
}

// rfxDequantizeAll applies the per-band quant shift to every subband including
// LL3, matching FreeRDP's rfx_quantization_decode (rfx_quantization.c line 57).
// Each band uses (q.X - 1) as the shift count, where X is HL1/LH1/HH1/HL2/...;
// values below 1 leave the band untouched.
func rfxDequantizeAll(coeffs []int16, q rfxQuant) {
	rfxShiftSubband(coeffs[0:1024], sub1(q.HL1))    // HL1
	rfxShiftSubband(coeffs[1024:2048], sub1(q.LH1)) // LH1
	rfxShiftSubband(coeffs[2048:3072], sub1(q.HH1)) // HH1
	rfxShiftSubband(coeffs[3072:3328], sub1(q.HL2)) // HL2
	rfxShiftSubband(coeffs[3328:3584], sub1(q.LH2)) // LH2
	rfxShiftSubband(coeffs[3584:3840], sub1(q.HH2)) // HH2
	rfxShiftSubband(coeffs[3840:3904], sub1(q.HL3)) // HL3
	rfxShiftSubband(coeffs[3904:3968], sub1(q.LH3)) // LH3
	rfxShiftSubband(coeffs[3968:4032], sub1(q.HH3)) // HH3
	rfxShiftSubband(coeffs[4032:4096], sub1(q.LL3)) // LL3
}

// rfxDequantizeSkipLL3 dequantises all bands except LL3, using each band's
// q.X value as the shift count *directly* (no implicit −1).  Caller is
// responsible for having applied the −1 if it's needed; non-progressive RFX
// uses rfxDequantizeSkipLL3FromWire which does it for you, while progressive's
// pre-computed shift quant already has the −1 baked in.  FreeRDP:
// progressive.c lines 880-898 (the shift quant has already been lsub'd by 1
// at line 1028).
func rfxDequantizeSkipLL3(coeffs []int16, shift rfxQuant) {
	rfxShiftSubband(coeffs[0:1024], shift.HL1)
	rfxShiftSubband(coeffs[1024:2048], shift.LH1)
	rfxShiftSubband(coeffs[2048:3072], shift.HH1)
	rfxShiftSubband(coeffs[3072:3328], shift.HL2)
	rfxShiftSubband(coeffs[3328:3584], shift.LH2)
	rfxShiftSubband(coeffs[3584:3840], shift.HH2)
	rfxShiftSubband(coeffs[3840:3904], shift.HL3)
	rfxShiftSubband(coeffs[3904:3968], shift.LH3)
	rfxShiftSubband(coeffs[3968:4032], shift.HH3)
}

// rfxDequantizeExtrapolateSkipLL3 dequantises every band except LL3 using the
// RFX_DWT_REDUCE_EXTRAPOLATE band layout (asymmetric band sizes).  Like
// rfxDequantizeSkipLL3 it applies the shift quant directly (the −1 is already
// baked into the progressive shift quant).  FreeRDP: progressive.c
// lines 903-919.  Band offsets/sizes:
//
//	HL1 0..1023, LH1 1023..2046, HH1 2046..3007,
//	HL2 3007..3279, LH2 3279..3551, HH2 3551..3807,
//	HL3 3807..3879, LH3 3879..3951, HH3 3951..4015.
func rfxDequantizeExtrapolateSkipLL3(coeffs []int16, shift rfxQuant) {
	rfxShiftSubband(coeffs[0:1023], shift.HL1)
	rfxShiftSubband(coeffs[1023:2046], shift.LH1)
	rfxShiftSubband(coeffs[2046:3007], shift.HH1)
	rfxShiftSubband(coeffs[3007:3279], shift.HL2)
	rfxShiftSubband(coeffs[3279:3551], shift.LH2)
	rfxShiftSubband(coeffs[3551:3807], shift.HH2)
	rfxShiftSubband(coeffs[3807:3879], shift.HL3)
	rfxShiftSubband(coeffs[3879:3951], shift.LH3)
	rfxShiftSubband(coeffs[3951:4015], shift.HH3)
}

// rfxDequantizeSkipLL3FromWire is the non-progressive variant that takes the
// raw wire quant and applies (q.X − 1) per band.  Used by rfx.go's tile
// pipeline.  FreeRDP: rfx_quantization.c lines 73-81.
func rfxDequantizeSkipLL3FromWire(coeffs []int16, q rfxQuant) {
	rfxShiftSubband(coeffs[0:1024], sub1(q.HL1))
	rfxShiftSubband(coeffs[1024:2048], sub1(q.LH1))
	rfxShiftSubband(coeffs[2048:3072], sub1(q.HH1))
	rfxShiftSubband(coeffs[3072:3328], sub1(q.HL2))
	rfxShiftSubband(coeffs[3328:3584], sub1(q.LH2))
	rfxShiftSubband(coeffs[3584:3840], sub1(q.HH2))
	rfxShiftSubband(coeffs[3840:3904], sub1(q.HL3))
	rfxShiftSubband(coeffs[3904:3968], sub1(q.LH3))
	rfxShiftSubband(coeffs[3968:4032], sub1(q.HH3))
}

// sub1 returns (v - 1) clamped to zero.  FreeRDP: rfx_quantization.c lines
// 73-82 pass `quantization_values[N] - 1` for every band.
func sub1(v uint8) uint8 {
	if v == 0 {
		return 0
	}
	return v - 1
}

// addQuant returns q1+q2 component-wise, used by progressive's bit-position
// arithmetic.  FreeRDP: progressive.c progressive_rfx_quant_add at line 81.
func addQuant(q1, q2 rfxQuant) rfxQuant {
	return rfxQuant{
		LL3: q1.LL3 + q2.LL3,
		LH3: q1.LH3 + q2.LH3,
		HL3: q1.HL3 + q2.HL3,
		HH3: q1.HH3 + q2.HH3,
		LH2: q1.LH2 + q2.LH2,
		HL2: q1.HL2 + q2.HL2,
		HH2: q1.HH2 + q2.HH2,
		LH1: q1.LH1 + q2.LH1,
		HL1: q1.HL1 + q2.HL1,
		HH1: q1.HH1 + q2.HH1,
	}
}

// subQuant returns q1-q2 component-wise saturated at zero.  FreeRDP:
// progressive.c progressive_rfx_quant_sub at line 143 (which returns FALSE
// instead of saturating; we follow the saturating route because an underflow
// here means "the encoder sent a malformed upgrade" which we want to skip,
// not abort).  Returns the diff and a flag indicating whether all components
// were non-negative.
func subQuant(q1, q2 rfxQuant) (rfxQuant, bool) {
	ok := true
	pick := func(a, b uint8) uint8 {
		if a < b {
			ok = false
			return 0
		}
		return a - b
	}
	return rfxQuant{
		LL3: pick(q1.LL3, q2.LL3),
		LH3: pick(q1.LH3, q2.LH3),
		HL3: pick(q1.HL3, q2.HL3),
		HH3: pick(q1.HH3, q2.HH3),
		LH2: pick(q1.LH2, q2.LH2),
		HL2: pick(q1.HL2, q2.HL2),
		HH2: pick(q1.HH2, q2.HH2),
		LH1: pick(q1.LH1, q2.LH1),
		HL1: pick(q1.HL1, q2.HL1),
		HH1: pick(q1.HH1, q2.HH1),
	}, ok
}

// quantLsub subtracts a scalar from each component, saturating at zero and
// returning whether all components were >= val (FreeRDP semantics:
// progressive.c progressive_rfx_quant_lsub at line 98).
func quantLsub(q rfxQuant, val uint8) (rfxQuant, bool) {
	ok := true
	pick := func(a uint8) uint8 {
		if a < val {
			ok = false
			return 0
		}
		return a - val
	}
	return rfxQuant{
		LL3: pick(q.LL3),
		LH3: pick(q.LH3),
		HL3: pick(q.HL3),
		HH3: pick(q.HH3),
		LH2: pick(q.LH2),
		HL2: pick(q.HL2),
		HH2: pick(q.HH2),
		LH1: pick(q.LH1),
		HL1: pick(q.HL1),
		HH1: pick(q.HH1),
	}, ok
}
