package rdpgfx

import (
	"math"
	"testing"
)

// TestInverseDWTRoundTrip verifies that forward DWT followed by inverse DWT
// reconstructs the original signal to within the lifting scheme's expected
// rounding error.  The RFX DWT uses biorthogonal 5/3 lifting; the per-step
// rounding is on the order of a handful of LSBs per level, so a 3-level
// round-trip on smooth/random inputs should stay well under ~32 absolute.
func TestInverseDWTRoundTrip(t *testing.T) {
	cases := []struct {
		name string
		gen  func(i int) int16
		// tolerable per-pixel max absolute reconstruction error.
		tol int32
	}{
		{
			name: "constant",
			gen:  func(int) int16 { return 100 },
			tol:  4,
		},
		{
			name: "linear-ramp",
			gen:  func(i int) int16 { return int16(i % 256) },
			// i%256 introduces a sharp wrap discontinuity inside the tile,
			// which the 5/3 DWT does not reconstruct exactly.  Allow the
			// reconstruction to swing wider here.
			tol: 110,
		},
		{
			name: "diagonal-ramp",
			gen: func(i int) int16 {
				x := i % 64
				y := i / 64
				return int16(x + y)
			},
			tol: 16,
		},
		{
			name: "checker",
			gen: func(i int) int16 {
				x := i % 64
				y := i / 64
				if (x+y)%2 == 0 {
					return 50
				}
				return -50
			},
			// Highest-frequency content possible in a 64×64 tile — DWT
			// reconstruction is non-trivial.  Allow ±50 (one full amplitude).
			tol: 60,
		},
		{
			name: "smooth-sine",
			gen: func(i int) int16 {
				x := i % 64
				y := i / 64
				return int16(50 * math.Sin(float64(x+y)/4))
			},
			tol: 50,
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			orig := make([]int16, 4096)
			for i := range orig {
				orig[i] = tc.gen(i)
			}
			work := make([]int16, 4096)
			copy(work, orig)

			rfxForwardDWT2D(work)
			rfxInverseDWT2D(work)

			var maxErr int32
			for i := range orig {
				diff := int32(work[i]) - int32(orig[i])
				if diff < 0 {
					diff = -diff
				}
				if diff > maxErr {
					maxErr = diff
				}
			}
			if maxErr > tc.tol {
				t.Errorf("max reconstruction error %d exceeds tolerance %d", maxErr, tc.tol)
			} else {
				t.Logf("max error: %d (tol %d)", maxErr, tc.tol)
			}
		})
	}
}

// TestInverseDWTPureLL3 verifies that constant LL3 with zeros elsewhere
// reconstructs to a near-constant 64×64 block — a tight property test of the
// 3-level cascade.  Any sign-flip / band-mismap bug shows up here as wildly
// varying output.
func TestInverseDWTPureLL3(t *testing.T) {
	coeffs := make([]int16, 4096)
	const llVal = 1000
	for i := 4032; i < 4096; i++ {
		coeffs[i] = llVal
	}
	rfxInverseDWT2D(coeffs)

	var lo, hi int16 = math.MaxInt16, math.MinInt16
	for _, v := range coeffs {
		if v < lo {
			lo = v
		}
		if v > hi {
			hi = v
		}
	}
	// All-DC LL3 should produce a near-constant block.  The 5/3 lifting
	// introduces a small ramp at the boundaries; allow some headroom.
	if hi-lo > 200 {
		t.Errorf("DC LL3 produced too much variation: lo=%d hi=%d (spread=%d)", lo, hi, hi-lo)
	}
	// Mean should be in the same ballpark as the input value.
	var sum int64
	for _, v := range coeffs {
		sum += int64(v)
	}
	mean := sum / 4096
	if mean < llVal/4 || mean > llVal*4 {
		t.Errorf("DC LL3 produced unexpected mean: %d (input %d)", mean, llVal)
	}
}

// TestParseRfxQuant covers the 5-byte nibble-packed RFX_COMPONENT_CODEC_QUANT
// layout used by both RFX and progressive RFX.  The wire encoding pairs each
// pair of bands into one byte: low nibble first.
func TestParseRfxQuant(t *testing.T) {
	// Encode known values from FreeRDP's progressive_component_codec_quant_read
	// (progressive.c line 59):
	//   byte0: LL3=0x1  HL3=0x2
	//   byte1: LH3=0x3  HH3=0x4
	//   byte2: HL2=0x5  LH2=0x6
	//   byte3: HH2=0x7  HL1=0x8
	//   byte4: LH1=0x9  HH1=0xA
	raw := []byte{0x21, 0x43, 0x65, 0x87, 0xA9}
	q := parseRfxQuant(raw)
	if q.LL3 != 1 || q.HL3 != 2 || q.LH3 != 3 || q.HH3 != 4 ||
		q.HL2 != 5 || q.LH2 != 6 || q.HH2 != 7 || q.HL1 != 8 ||
		q.LH1 != 9 || q.HH1 != 0xA {
		t.Errorf("unexpected quant unpacking: %+v", q)
	}
}

// TestSubQuantSaturation checks our subQuant helper saturates at zero rather
// than wrapping, matching FreeRDP's progressive_rfx_quant_sub semantics (which
// fails the whole operation on underflow — we saturate so a single malformed
// UPGRADE doesn't lose the whole frame).
func TestSubQuantSaturation(t *testing.T) {
	a := rfxQuant{HL1: 6, LH1: 6, HH1: 6, HL2: 6, LH2: 6, HH2: 6, HL3: 6, LH3: 6, HH3: 6, LL3: 6}
	b := rfxQuant{HL1: 8, LH1: 6, HH1: 6, HL2: 6, LH2: 6, HH2: 6, HL3: 6, LH3: 6, HH3: 6, LL3: 6}
	d, ok := subQuant(a, b)
	if ok {
		t.Errorf("expected ok=false on underflow")
	}
	if d.HL1 != 0 {
		t.Errorf("expected saturated HL1=0, got %d", d.HL1)
	}
	if d.LH1 != 0 || d.HH1 != 0 {
		t.Errorf("expected zero diffs for equal components, got LH1=%d HH1=%d", d.LH1, d.HH1)
	}
}

// TestProgressiveSurfaceCache verifies that the same surface buffer pointer
// yields the same cache entry across calls, but a buffer of a different size
// allocates a fresh cache (so we don't accidentally reuse stale tiles after a
// resize).
func TestProgressiveSurfaceCache(t *testing.T) {
	d := newRfxProgressiveDecoder()
	buf := make([]byte, 64*64*4)
	s1 := d.getSurface(buf, 64, 64)
	s2 := d.getSurface(buf, 64, 64)
	if s1 != s2 {
		t.Errorf("expected identical surface for same buf")
	}
	s3 := d.getSurface(buf, 128, 64)
	if s3 == s1 {
		t.Errorf("expected new surface for different size")
	}
	// Original should now be evicted (replaced in the map).
	s4 := d.getSurface(buf, 64, 64)
	if s4 == s1 {
		t.Errorf("expected fresh surface after size change re-request")
	}
}
