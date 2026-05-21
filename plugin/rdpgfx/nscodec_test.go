package rdpgfx

import (
	"bytes"
	"encoding/binary"
	"testing"
)

// buildNSCPayload assembles an NSCodec payload from already-RLE'd plane
// data (or raw planes — pass planeByteCount == len(rawPlanes[i]) for an
// uncompressed plane and orgByteCount == that same length).
func buildNSCPayload(planes [4][]byte, colorLoss, chromaSub uint8) []byte {
	var buf bytes.Buffer
	for i := 0; i < 4; i++ {
		var tmp [4]byte
		binary.LittleEndian.PutUint32(tmp[:], uint32(len(planes[i])))
		buf.Write(tmp[:])
	}
	buf.WriteByte(colorLoss)
	buf.WriteByte(chromaSub)
	buf.WriteByte(0)
	buf.WriteByte(0)
	for i := 0; i < 4; i++ {
		buf.Write(planes[i])
	}
	return buf.Bytes()
}

// TestNSCDecode_NoSubsample_NoRLE feeds a known YCoCg sample through
// the decoder with chromaSubsampling=0 and uncompressed planes, then
// checks the BGRA output matches the FreeRDP-spec inverse transform.
func TestNSCDecode_NoSubsample_NoRLE(t *testing.T) {
	const w, h = 4, 2
	// Y=128 across, Co=10, Cg=-5 (encoded as byte 0xFB), A=0xFF.
	// With ColorLossLevel=1 (shift=0), Co'=10, Cg'=-5.
	// R = Y + Co - Cg = 128 + 10 - (-5) = 143
	// G = Y + Cg       = 128 + (-5)    = 123
	// B = Y - Co - Cg  = 128 - 10 - (-5) = 123
	y := bytes.Repeat([]byte{128}, w*h)
	co := bytes.Repeat([]byte{10}, w*h)
	cg := bytes.Repeat([]byte{0xFB}, w*h) // -5 as signed
	a := bytes.Repeat([]byte{0xFF}, w*h)
	payload := buildNSCPayload([4][]byte{y, co, cg, a}, 1, 0)

	dec := newNSCodec()
	surfStride := w * 4
	out := make([]byte, surfStride*h)
	if !dec.Decode(payload, w, h, out, surfStride, 0, 0) {
		t.Fatal("Decode failed")
	}
	for i := 0; i < w*h; i++ {
		b := out[i*4]
		g := out[i*4+1]
		r := out[i*4+2]
		av := out[i*4+3]
		if b != 123 || g != 123 || r != 143 || av != 0xFF {
			t.Errorf("pixel %d: got BGRA=(%d,%d,%d,%d), want (123,123,143,255)",
				i, b, g, r, av)
		}
	}
}

// TestNSCDecode_ColorLossShift verifies the colorloss recovery shift.
// ColorLossLevel=3 -> shift=2.  Co byte value = 5 -> Co' = int8(5<<2) = 20.
func TestNSCDecode_ColorLossShift(t *testing.T) {
	const w, h = 2, 1
	y := []byte{100, 100}
	co := []byte{5, 5}   // -> 20
	cg := []byte{0xFE, 0xFE} // (-2) << 2 with int8 truncation: 0xFE<<2 = 0xF8 = -8 as int8
	a := []byte{0xFF, 0xFF}
	payload := buildNSCPayload([4][]byte{y, co, cg, a}, 3, 0)

	dec := newNSCodec()
	surfStride := w * 4
	out := make([]byte, surfStride*h)
	if !dec.Decode(payload, w, h, out, surfStride, 0, 0) {
		t.Fatal("Decode failed")
	}
	// R = 100 + 20 - (-8) = 128
	// G = 100 + (-8)      = 92
	// B = 100 - 20 - (-8) = 88
	for i := 0; i < w; i++ {
		b := out[i*4]
		g := out[i*4+1]
		r := out[i*4+2]
		if b != 88 || g != 92 || r != 128 {
			t.Errorf("pixel %d: got BGR=(%d,%d,%d), want (88,92,128)", i, b, g, r)
		}
	}
}

// TestNSCDecode_Subsample verifies chromaSubsampling=1 plane sizes and
// the (x>>1, y>>1) indexing.
func TestNSCDecode_Subsample(t *testing.T) {
	const w, h = 4, 2
	// width=4 rounds up to tempWidth=8 (already), height=2 rounds up to 2.
	// tempWidth=8, so Y plane = 8*2=16 bytes; Co/Cg plane = 4*1=4 bytes.
	// A plane = w*h = 8 bytes.
	tempW := 8
	y := make([]byte, tempW*h)
	for i := range y {
		y[i] = 128
	}
	co := []byte{10, 10, 10, 10}     // 4 samples covering 4 columns when subsampled
	cg := []byte{0xFB, 0xFB, 0xFB, 0xFB} // -5
	a := bytes.Repeat([]byte{0xFF}, w*h)
	payload := buildNSCPayload([4][]byte{y, co, cg, a}, 1, 1)

	dec := newNSCodec()
	surfStride := w * 4
	out := make([]byte, surfStride*h)
	if !dec.Decode(payload, w, h, out, surfStride, 0, 0) {
		t.Fatal("Decode failed")
	}
	for i := 0; i < w*h; i++ {
		b := out[i*4]
		g := out[i*4+1]
		r := out[i*4+2]
		if b != 123 || g != 123 || r != 143 {
			t.Errorf("pixel %d: got BGR=(%d,%d,%d), want (123,123,143)", i, b, g, r)
		}
	}
}

// TestNSCRLEDecode_Literal verifies a stream with no runs (every byte
// differs from its successor) plus the mandatory 4-byte tail.
func TestNSCRLEDecode_Literal(t *testing.T) {
	// originalSize=8; bytes: 1,2,3,4 + literal tail 5,6,7,8 = no runs.
	// FreeRDP loop runs while left>4, so left goes 8->7->6->5->4 (4 bytes
	// literal), then the tail copy emits the last 4.
	in := []byte{1, 2, 3, 4, 5, 6, 7, 8}
	out := make([]byte, 16)
	if !nscRLEDecode(in, out, 8) {
		t.Fatal("RLE decode failed")
	}
	expect := []byte{1, 2, 3, 4, 5, 6, 7, 8}
	if !bytes.Equal(out[:8], expect) {
		t.Errorf("got %v, want %v", out[:8], expect)
	}
}

// TestNSCRLEDecode_ShortRun encodes a run of 7 identical bytes (0xAA)
// followed by a 4-byte tail.  Encoding: 0xAA 0xAA 0x05 ... means value
// 0xAA, duplicate, count=5 -> total run length 7.  Then the 4-byte
// tail follows.
func TestNSCRLEDecode_ShortRun(t *testing.T) {
	// originalSize=11: 7 bytes of 0xAA + 4-byte tail [1,2,3,4].
	in := []byte{0xAA, 0xAA, 0x05, 0x01, 0x02, 0x03, 0x04}
	out := make([]byte, 16)
	if !nscRLEDecode(in, out, 11) {
		t.Fatal("RLE decode failed")
	}
	expect := []byte{0xAA, 0xAA, 0xAA, 0xAA, 0xAA, 0xAA, 0xAA, 1, 2, 3, 4}
	if !bytes.Equal(out[:11], expect) {
		t.Errorf("got %v, want %v", out[:11], expect)
	}
}

// TestNSCRLEDecode_LongRun encodes a run >256 using the 0xFF extended
// uint32 form.  Run length = 300; tail [9,9,9,9].
func TestNSCRLEDecode_LongRun(t *testing.T) {
	// 0x5A 0x5A 0xFF <u32 LE 300> tail
	in := []byte{0x5A, 0x5A, 0xFF, 0x2C, 0x01, 0x00, 0x00, 9, 9, 9, 9}
	out := make([]byte, 304)
	if !nscRLEDecode(in, out, 304) {
		t.Fatal("RLE decode failed")
	}
	for i := 0; i < 300; i++ {
		if out[i] != 0x5A {
			t.Fatalf("at %d got %#x want 0x5A", i, out[i])
		}
	}
	for i := 300; i < 304; i++ {
		if out[i] != 9 {
			t.Fatalf("at %d got %#x want 9", i, out[i])
		}
	}
}
