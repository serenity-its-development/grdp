package rdpgfx

import (
	"encoding/binary"
	"strings"
	"testing"
)

func TestCodingStatsReadsAVC420Regions(t *testing.T) {
	TakeCodingStats() // start clean
	// two regions: 100x100 at QP 20 (progressive, quality 90) and 300x100 at QP 32 (quality 50)
	b := make([]byte, 4+2*8+2*2+10)
	binary.LittleEndian.PutUint32(b, 2)
	rect := func(i int, l, tp, r, bt uint16) {
		o := 4 + i*8
		binary.LittleEndian.PutUint16(b[o:], l)
		binary.LittleEndian.PutUint16(b[o+2:], tp)
		binary.LittleEndian.PutUint16(b[o+4:], r)
		binary.LittleEndian.PutUint16(b[o+6:], bt)
	}
	rect(0, 0, 0, 100, 100)
	rect(1, 100, 0, 400, 100)
	b[20], b[21] = 20|0x80, 90
	b[22], b[23] = 32, 50
	noteSurfaceCommand(codecAVC420, b)
	noteSurfaceCommand(codecClearCodec, make([]byte, 2048))
	noteDroppedFrame()
	got := TakeCodingStats()
	for _, want := range []string{"avc420=1/0kB", "clear=1/2kB", "avc_qp=29.0(min20,max32)", "avc_quality=60", "progressive=25%", "qp_fine/mid/coarse=25/0/75%", "dropped=1"} {
		if !strings.Contains(got, want) {
			t.Fatalf("stats %q lack %q", got, want)
		}
	}
	if TakeCodingStats() != "" {
		t.Fatal("stats not reset")
	}
	noteSurfaceCommand(codecAVC420, []byte{9, 0, 0, 0, 1}) // truncated metablock: counted, not read
	if got := TakeCodingStats(); !strings.Contains(got, "avc420=1/") || strings.Contains(got, "avc_qp") {
		t.Fatalf("stats %q", got)
	}
}
