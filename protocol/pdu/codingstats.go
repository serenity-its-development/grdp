package pdu

// codingstats.go — how Windows sends the screen outside the graphics pipeline (a measurement,
// 2026-11-03). The gateway gets no RDPGFX at all (rdpgfx codingstats.go stayed empty): the screen
// comes as surface bits, by codec (0 raw, 1 NSCodec, 3 RemoteFX: both lossy), or as bitmap updates
// (lossless RLE / planar; planar's header byte says whether colour was reduced: CLL bits 0-2, chroma
// subsampling bit 3, MS-RDPEGDI 2.2.2.5.1). Read and reset with TakeClassicStats.

import (
	"fmt"
	"sort"
	"strings"
	"sync"
)

var classic struct {
	sync.Mutex
	surf      map[uint8]*classicCount // surface bits by codec
	nsc       map[string]int          // NSCodec "colour loss level/chroma subsampling"
	bmp       map[string]*classicCount // bitmap updates by "bpp/raw|rle|planar"
	planarCLL map[uint8]int
}

type classicCount struct {
	n, bytes, pixels int
}

func (c *classicCount) add(bytes, pixels int) {
	c.n++
	c.bytes += bytes
	c.pixels += pixels
}

// noteSurfaceBits counts a SET/STREAM_SURFACE_BITS command; for NSCodec also its colour loss level
// and chroma subsampling (MS-RDPNSC 2.2.1: bytes 16 and 17 of the bitmap stream).
func noteSurfaceBits(codec uint8, data []byte, w, h int) {
	classic.Lock()
	defer classic.Unlock()
	if classic.surf == nil {
		classic.surf = map[uint8]*classicCount{}
		classic.nsc = map[string]int{}
	}
	if codec == 1 && len(data) >= 18 {
		classic.nsc[fmt.Sprintf("%d/%d", data[16], data[17])]++
	}
	bytes := len(data)
	c := classic.surf[codec]
	if c == nil {
		c = &classicCount{}
		classic.surf[codec] = c
	}
	c.add(bytes, w*h)
}

// NoteBitmap counts a bitmap update rectangle (grdp.go), data as the server sent it.
func NoteBitmap(bpp uint16, compressed bool, data []byte, w, h int) {
	classic.Lock()
	defer classic.Unlock()
	if classic.bmp == nil {
		classic.bmp = map[string]*classicCount{}
		classic.planarCLL = map[uint8]int{}
	}
	kind := "raw"
	if compressed {
		kind = "rle"
		if bpp == 32 && len(data) > 0 {
			kind = "planar"
			classic.planarCLL[data[0]&0x0f]++ // colour loss level and chroma subsampling bit
		}
	}
	key := fmt.Sprintf("%dbpp/%s", bpp, kind)
	c := classic.bmp[key]
	if c == nil {
		c = &classicCount{}
		classic.bmp[key] = c
	}
	c.add(len(data), w*h)
}

// TakeClassicStats reports what came since the last call and starts a new count ("" if nothing).
func TakeClassicStats() string {
	classic.Lock()
	defer classic.Unlock()
	var sb strings.Builder
	names := map[uint8]string{0: "raw", 1: "nscodec", 3: "remotefx"}
	var codecs []int
	for c := range classic.surf {
		codecs = append(codecs, int(c))
	}
	sort.Ints(codecs)
	for _, c := range codecs {
		n := classic.surf[uint8(c)]
		name := names[uint8(c)]
		if name == "" {
			name = fmt.Sprintf("codec%d", c)
		}
		fmt.Fprintf(&sb, "surf_%s=%d/%dkB/%dkpx ", name, n.n, (n.bytes+512)/1024, n.pixels/1000)
	}
	var kinds []string
	for k := range classic.bmp {
		kinds = append(kinds, k)
	}
	sort.Strings(kinds)
	for _, k := range kinds {
		n := classic.bmp[k]
		fmt.Fprintf(&sb, "bmp_%s=%d/%dkB/%dkpx ", k, n.n, (n.bytes+512)/1024, n.pixels/1000)
	}
	if len(classic.nsc) > 0 {
		var ks []string
		for k := range classic.nsc {
			ks = append(ks, k)
		}
		sort.Strings(ks)
		sb.WriteString("nsc_colourloss/chroma=")
		for i, k := range ks {
			if i > 0 {
				sb.WriteString(",")
			}
			fmt.Fprintf(&sb, "%s:%d", k, classic.nsc[k])
		}
		sb.WriteString(" ")
	}
	if len(classic.planarCLL) > 0 {
		var hs []int
		for h := range classic.planarCLL {
			hs = append(hs, int(h))
		}
		sort.Ints(hs)
		sb.WriteString("planar_cll_cs=")
		for i, h := range hs {
			if i > 0 {
				sb.WriteString(",")
			}
			fmt.Fprintf(&sb, "%d:%d", h, classic.planarCLL[uint8(h)])
		}
	}
	classic.surf, classic.bmp, classic.planarCLL, classic.nsc = nil, nil, nil, nil
	return strings.TrimSpace(sb.String())
}
