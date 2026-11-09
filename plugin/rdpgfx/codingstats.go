package rdpgfx

// codingstats.go — how hard Windows compresses the screen it sends us (a measurement, 2026-11-03).
//
// Every surface command is counted by codec with its size. For H.264 (AVC420) the metablock in
// front of the bitstream (MS-RDPEGFX 2.2.4.4.1 RDPGFX_AVC420_METABLOCK) carries, per region, the
// quantisation parameter Windows encoded it at (qpVal bits 0-5: lower = finer, 51 = coarsest), a
// "progressive" bit (bit 7) and a quality value 0-100 (RDPGFX_H264_QUANT_QUALITY). They are read,
// weighted by region area, and reported with TakeCodingStats. Nothing here changes what is decoded.

import (
	"encoding/binary"
	"fmt"
	"sort"
	"strings"
	"sync"
)

var coding struct {
	sync.Mutex
	cmds    map[uint16]int
	bytes   map[uint16]int
	dropped int
	// AVC420 regions, weighted by area (pixels)
	regions  int
	area     float64
	qpArea   float64
	qpMin    int
	qpMax    int
	qualArea float64
	progArea float64
	// regions by QP band, by area: fine (<=22), mid (23-29), coarse (>=30)
	fineArea, midArea, coarseArea float64
	// RemoteFX (MS-RDPRFX) tilesets: tiles, and their quantisation tables (LL3 LH3 HL3 HH3 LH2 HL2
	// HH2 LH1 HL1 HH1, 6 = finest Windows uses, higher = coarser)
	rfxSets, rfxTiles int
	rfxQuant          map[string]int
}

// noteRFXTileset counts a RemoteFX tileset and its quantisation tables.
func noteRFXTileset(tiles int, quants []rfxQuant) {
	coding.Lock()
	defer coding.Unlock()
	coding.rfxSets++
	coding.rfxTiles += tiles
	if coding.rfxQuant == nil {
		coding.rfxQuant = map[string]int{}
	}
	for _, q := range quants {
		coding.rfxQuant[fmt.Sprintf("%d.%d.%d.%d.%d.%d.%d.%d.%d.%d", q.LL3, q.LH3, q.HL3, q.HH3, q.LH2, q.HL2, q.HH2, q.LH1, q.HL1, q.HH1)]++
	}
}

func codecName(id uint16) string {
	switch id {
	case codecUncompressed:
		return "raw"
	case codecCaVideo:
		return "remotefx"
	case codecClearCodec:
		return "clear"
	case codecPlanar:
		return "planar"
	case codecAVC420:
		return "avc420"
	case codecAVC444:
		return "avc444"
	case codecAVC444v2:
		return "avc444v2"
	case codecProgressive:
		return "progressive"
	}
	return fmt.Sprintf("codec%#x", id)
}

// noteSurfaceCommand counts one wire-to-surface command of codec id with its bitmap data.
func noteSurfaceCommand(id uint16, bmp []byte) {
	coding.Lock()
	defer coding.Unlock()
	if coding.cmds == nil {
		coding.cmds = map[uint16]int{}
		coding.bytes = map[uint16]int{}
	}
	coding.cmds[id]++
	coding.bytes[id] += len(bmp)
	if id == codecAVC420 {
		noteAVC420Meta(bmp)
	}
}

// noteDroppedFrame counts a frame the decode queue had no room for.
func noteDroppedFrame() {
	coding.Lock()
	coding.dropped++
	coding.Unlock()
}

// noteAVC420Meta reads the region rectangles and their quant/quality values. coding is locked.
func noteAVC420Meta(b []byte) {
	if len(b) < 4 {
		return
	}
	n := int(binary.LittleEndian.Uint32(b))
	if n <= 0 || n > 65536 || 4+n*10 > len(b) {
		return
	}
	for i := 0; i < n; i++ {
		r := b[4+i*8:]
		left := int(binary.LittleEndian.Uint16(r[0:]))
		top := int(binary.LittleEndian.Uint16(r[2:]))
		right := int(binary.LittleEndian.Uint16(r[4:]))
		bottom := int(binary.LittleEndian.Uint16(r[6:]))
		a := float64((right - left) * (bottom - top))
		if a <= 0 {
			a = 1
		}
		q := b[4+n*8+i*2:]
		qp := int(q[0] & 0x3f)
		prog := q[0]&0x80 != 0
		quality := float64(q[1])
		if coding.regions == 0 || qp < coding.qpMin {
			coding.qpMin = qp
		}
		if coding.regions == 0 || qp > coding.qpMax {
			coding.qpMax = qp
		}
		coding.regions++
		coding.area += a
		coding.qpArea += float64(qp) * a
		coding.qualArea += quality * a
		if prog {
			coding.progArea += a
		}
		switch {
		case qp <= 22:
			coding.fineArea += a
		case qp <= 29:
			coding.midArea += a
		default:
			coding.coarseArea += a
		}
	}
}

// TakeCodingStats reports what Windows sent since the last call and starts a new count. Empty when
// nothing came (a still screen).
func TakeCodingStats() string {
	coding.Lock()
	defer coding.Unlock()
	if len(coding.cmds) == 0 && coding.dropped == 0 && coding.rfxSets == 0 {
		return ""
	}
	var ids []int
	for id := range coding.cmds {
		ids = append(ids, int(id))
	}
	sort.Ints(ids)
	var sb strings.Builder
	for _, id := range ids {
		fmt.Fprintf(&sb, "%s=%d/%dkB ", codecName(uint16(id)), coding.cmds[uint16(id)], (coding.bytes[uint16(id)]+512)/1024)
	}
	if coding.regions > 0 {
		a := coding.area
		fmt.Fprintf(&sb, "avc_qp=%.1f(min%d,max%d) avc_quality=%.0f progressive=%.0f%% qp_fine/mid/coarse=%.0f/%.0f/%.0f%% ",
			coding.qpArea/a, coding.qpMin, coding.qpMax, coding.qualArea/a, 100*coding.progArea/a,
			100*coding.fineArea/a, 100*coding.midArea/a, 100*coding.coarseArea/a)
	}
	if coding.dropped > 0 {
		fmt.Fprintf(&sb, "dropped=%d ", coding.dropped)
	}
	if coding.rfxSets > 0 {
		fmt.Fprintf(&sb, "rfx_tilesets=%d rfx_tiles=%d rfx_quant=", coding.rfxSets, coding.rfxTiles)
		var qs []string
		for q := range coding.rfxQuant {
			qs = append(qs, q)
		}
		sort.Strings(qs)
		for i, q := range qs {
			if i > 0 {
				sb.WriteString(",")
			}
			fmt.Fprintf(&sb, "%s:%d", q, coding.rfxQuant[q])
		}
		sb.WriteString(" ")
	}
	coding.rfxSets, coding.rfxTiles, coding.rfxQuant = 0, 0, nil
	coding.cmds, coding.bytes, coding.dropped = nil, nil, 0
	coding.regions, coding.area, coding.qpArea, coding.qualArea, coding.progArea = 0, 0, 0, 0, 0
	coding.fineArea, coding.midArea, coding.coarseArea = 0, 0, 0
	return strings.TrimSpace(sb.String())
}
