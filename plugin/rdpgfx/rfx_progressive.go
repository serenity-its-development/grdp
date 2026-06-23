package rdpgfx

// RFX Progressive Codec decoder (MS-RDPRFX / MS-RDPEGFX 2.2.4).
// Handles RDPGFX_CODECID_CAPROGRESSIVE (0x0009) in WIRE_TO_SURFACE_PDU_2.
//
// This file is a faithful Go port of FreeRDP's libfreerdp/codec/progressive.c
// (canonical C reference at /tmp/freerdp-ref/progressive.c in this work tree).
// Helpers carry a `// FreeRDP: progressive.c line N` comment pointing back to
// the C function each one mirrors.
//
// Entry point lives in this file; the per-tile decode (FIRST/SIMPLE/UPGRADE),
// quantization helpers, and inverse DWT live in:
//   rfx_progressive_tile.go  — tile state cache + decode dispatch
//   rfx_quantization.go      — per-band scalar quantization
//   rfx_dwt.go               — 3-level inverse 2D DWT (non-extrapolate)
//   rfx_dwt_extrapolate.go   — 3-level inverse 2D DWT (RFX_DWT_REDUCE_EXTRAPOLATE)
//
// Both band layouts are supported: the legacy 1024/256/64 layout and the
// RFX_DWT_REDUCE_EXTRAPOLATE asymmetric 1023/1023/961…81 layout, selected per
// region by the RFX_DWT_REDUCE_EXTRAPOLATE flag and threaded through the tile
// decode path as `extrapolate`.

import (
	"bytes"
	"fmt"
	"log/slog"
	"runtime"
	"sync"
	"unsafe"

	"github.com/nakagami/grdp/core"
)

// Progressive block types — different from non-progressive WBT_* at same values.
// FreeRDP: rfx_constants.h lines 42-49.
const (
	progWBTSync        = 0xCCC0
	progWBTFrameBegin  = 0xCCC1
	progWBTFrameEnd    = 0xCCC2
	progWBTContext     = 0xCCC3
	progWBTRegion      = 0xCCC4
	progWBTTileSimple  = 0xCCC5
	progWBTTileFirst   = 0xCCC6
	progWBTTileUpgrade = 0xCCC7
)

const rfxTileSize = 64

// Region/context/tile flags.  Values are from FreeRDP's rfx_types.h.
const (
	rfxSubbandDiffing       = 0x01 // PROGRESSIVE_BLOCK_CONTEXT.flags
	rfxTileDifference       = 0x01 // RFX_PROGRESSIVE_TILE.flags
	rfxDwtReduceExtrapolate = 0x01 // PROGRESSIVE_BLOCK_REGION.flags
)

// rfxQuant holds the 10 quantization values for one component (5 bytes, 10
// nibbles).  FreeRDP packs the same way; see progressive.c
// progressive_component_codec_quant_read at line 59.
type rfxQuant struct {
	LL3, LH3, HL3, HH3 uint8
	LH2, HL2, HH2      uint8
	LH1, HL1, HH1      uint8
}

// rfxProgQuant is one of the per-quality progressive quantization triples
// supplied in the REGION block.  FreeRDP: RFX_PROGRESSIVE_CODEC_QUANT.
type rfxProgQuant struct {
	quality uint8
	y       rfxQuant
	cb      rfxQuant
	cr      rfxQuant
}

// rfxRect represents a rectangle of decoded tiles.
type rfxRect struct {
	x, y, w, h int
}

// rfxProgressiveDecoder is the public type retained for backward compatibility
// with the dispatcher in rdpgfx.go.  All per-surface state lives in surfaces,
// keyed by the surface buffer's backing-array address.  Concurrent Decode
// calls on different surfaces are safe; concurrent calls on the same surface
// are serialised by mu (real RDPGFX dispatch is single-threaded for a given
// surface but defence in depth is cheap).
type rfxProgressiveDecoder struct {
	mu       sync.Mutex
	surfaces map[uintptr]*progressiveSurface
}

func newRfxProgressiveDecoder() *rfxProgressiveDecoder {
	return &rfxProgressiveDecoder{
		surfaces: make(map[uintptr]*progressiveSurface),
	}
}

// progressiveSurface holds the per-tile coefficient cache for one RDP surface.
// FreeRDP keeps an equivalent PROGRESSIVE_SURFACE_CONTEXT keyed by surfaceId;
// our dispatcher hands us a raw surfData []byte so we use its backing-array
// address as identity.  width/height are recorded so that a buffer reuse for
// a different surface size invalidates the cache.
//
// tiles is a flat grid sized (gridW * gridH) of *progressiveTile, allocated on
// first use; entries are lazy-initialised so unused tiles don't pay the cost.
type progressiveSurface struct {
	width, height int
	gridW, gridH  int
	tiles         []*progressiveTile
}

// progressiveTile holds the cached coefficient state for one 64×64 tile.
// Following FreeRDP's RFX_PROGRESSIVE_TILE:
//
//	current[0..3] — post-shift coefficients per component, refined across
//	                UPGRADE passes (4096 int16 per component).
//	sign[0..3]    — pre-shift signed-magnitude coefficients captured on the
//	                FIRST pass; UPGRADE consults sign[i] to decide whether a
//	                non-LL refinement bit comes from the RAW or SRL stream.
//	yBitPos/...   — cumulative (quant + progQuant) bitpos per component, used
//	                by UPGRADE to compute the delta bit-count vs the previous
//	                pass.  FreeRDP: progressive.c line 1024.
//	yQuant/...    — the non-progressive quant snapshot from the FIRST pass.
//	                Used by UPGRADE to detect server-side quant changes.
type progressiveTile struct {
	xIdx, yIdx uint16
	pass       int   // 1 after FIRST/SIMPLE, +1 on each UPGRADE
	allocated  bool  // false until the first FIRST sets up state

	current [3][4096]int16
	sign    [3][4096]int16

	yBitPos, cbBitPos, crBitPos rfxQuant
	yQuant, cbQuant, crQuant    rfxQuant
}

// surfaceKey returns the cache key for the given surface buffer.  Backed by
// the slice's data pointer; safe because grdp re-uses one slice for a
// surface's lifetime.  width/height are checked against the cached entry to
// invalidate on size change (or buffer reuse for a different surface).
func surfaceKey(surfData []byte) uintptr {
	if len(surfData) == 0 {
		return 0
	}
	return uintptr(unsafe.Pointer(&surfData[0]))
}

func (d *rfxProgressiveDecoder) getSurface(surfData []byte, w, h int) *progressiveSurface {
	key := surfaceKey(surfData)
	if key == 0 {
		return nil
	}
	d.mu.Lock()
	defer d.mu.Unlock()
	s, ok := d.surfaces[key]
	if !ok || s.width != w || s.height != h {
		gridW := (w + rfxTileSize - 1) / rfxTileSize
		gridH := (h + rfxTileSize - 1) / rfxTileSize
		s = &progressiveSurface{
			width:  w,
			height: h,
			gridW:  gridW,
			gridH:  gridH,
			tiles:  make([]*progressiveTile, gridW*gridH),
		}
		d.surfaces[key] = s
	}
	return s
}

// getTile returns the tile slot for (xIdx, yIdx), allocating it on first use.
// Returns nil if the indices fall outside the current grid.
func (s *progressiveSurface) getTile(xIdx, yIdx int) *progressiveTile {
	if xIdx < 0 || yIdx < 0 || xIdx >= s.gridW || yIdx >= s.gridH {
		return nil
	}
	idx := yIdx*s.gridW + xIdx
	t := s.tiles[idx]
	if t == nil {
		t = &progressiveTile{xIdx: uint16(xIdx), yIdx: uint16(yIdx)}
		s.tiles[idx] = t
	}
	return t
}

// Decode processes RFX Progressive codec data, rendering tiles onto the
// provided surface buffer.  Returns the bounding rectangles of decoded regions
// (one entry per RFX_RECT in each REGION block).  FreeRDP: progressive.c
// progressive_decompress at line 2418.
func (d *rfxProgressiveDecoder) Decode(data []byte, surfData []byte, width, height int) []rfxRect {
	if len(data) == 0 || len(surfData) == 0 {
		return nil
	}
	surface := d.getSurface(surfData, width, height)
	if surface == nil {
		return nil
	}

	var rects []rfxRect
	var ctxFlags uint8

	r := bytes.NewReader(data)
	for r.Len() >= 6 {
		blockType, err := core.ReadUint16LE(r)
		if err != nil {
			break
		}
		blockLen, err := core.ReadUInt32LE(r)
		if err != nil {
			break
		}
		if blockLen < 6 {
			slog.Debug("RFX progressive: bad block length", "blockType", fmt.Sprintf("0x%04X", blockType), "blockLen", blockLen)
			break
		}
		payloadLen := int(blockLen) - 6
		if r.Len() < payloadLen {
			slog.Debug("RFX progressive: truncated block", "blockType", fmt.Sprintf("0x%04X", blockType), "want", payloadLen, "have", r.Len())
			break
		}
		// Slice the payload directly so sub-block decoders can index without
		// reallocating; the bytes.Reader keeps the cursor advancing for the
		// next outer iteration.
		off, _ := r.Seek(0, 1) // current pos
		payload := data[off : off+int64(payloadLen)]
		_, _ = r.Seek(int64(payloadLen), 1)

		switch blockType {
		case progWBTSync:
			// magic + version; nothing to validate beyond what FreeRDP does.
		case progWBTFrameBegin, progWBTFrameEnd:
			// frame begin: frameIndex(4) + regionCount(2)
			// frame end: empty
			// nothing surface-affecting here.
		case progWBTContext:
			// ctxId(1) + tileSize(2) + flags(1)
			if len(payload) >= 4 {
				ctxFlags = payload[3]
			}
		case progWBTRegion:
			rrects := d.parseRegion(payload, surface, surfData, width, height, ctxFlags)
			rects = append(rects, rrects...)
		default:
			slog.Debug("RFX progressive: unknown block", "type", fmt.Sprintf("0x%04X", blockType))
		}
	}

	return rects
}

// parseRegion handles PROGRESSIVE_WBT_REGION (0xCCC4).  Layout (12-byte fixed
// header + variable arrays + tile sub-blocks):
//
//	tileSize(1) numRects(2) numQuant(1) numProgQuant(1) flags(1)
//	numTiles(2) tileDataSize(4)
//	rects[numRects]            — each 8 bytes (x, y, w, h as u16)
//	quantVals[numQuant]        — each 5 bytes (10 nibbles)
//	progQuantVals[numProgQuant]— each 1 byte quality + 3×5 bytes Y/Cb/Cr quant
//	tiles                      — TILE_SIMPLE / TILE_FIRST / TILE_UPGRADE blocks
//
// FreeRDP: progressive.c progressive_wb_region at line 2129 (header parsing)
// + progressive_process_tiles at line 1684 (tile dispatch).
func (d *rfxProgressiveDecoder) parseRegion(data []byte, surface *progressiveSurface, surfData []byte, outW, outH int, ctxFlags uint8) []rfxRect {
	if len(data) < 12 {
		return nil
	}
	r := bytes.NewReader(data)
	tileSize, _ := core.ReadUInt8(r)
	if tileSize != 64 {
		slog.Debug("RFX progressive: unexpected tileSize", "tileSize", tileSize)
		// Continue anyway; FreeRDP errors out but we'd rather try.
	}
	numRects, _ := core.ReadUint16LE(r)
	numQuant, _ := core.ReadUInt8(r)
	numProgQuant, _ := core.ReadUInt8(r)
	regionFlags, _ := core.ReadUInt8(r)
	_, _ = core.ReadUint16LE(r) // numTiles, FreeRDP only uses for sanity
	_, _ = core.ReadUInt32LE(r) // tileDataSize, ditto

	// RFX_DWT_REDUCE_EXTRAPOLATE selects the asymmetric extrapolate band
	// layout (1023/1023/961 … 81) and the matching extrapolate inverse DWT.
	// FreeRDP: progressive.c line 959 (extrapolate = region->flags &
	// RFX_DWT_REDUCE_EXTRAPOLATE).  We support it for FIRST/SIMPLE tiles.
	extrapolate := (regionFlags & rfxDwtReduceExtrapolate) != 0

	rects := make([]rfxRect, 0, numRects)
	for i := 0; i < int(numRects); i++ {
		x, err := core.ReadUint16LE(r)
		if err != nil {
			return nil
		}
		y, _ := core.ReadUint16LE(r)
		w, _ := core.ReadUint16LE(r)
		h, _ := core.ReadUint16LE(r)
		rects = append(rects, rfxRect{x: int(x), y: int(y), w: int(w), h: int(h)})
	}

	quants := make([]rfxQuant, numQuant)
	for i := 0; i < int(numQuant); i++ {
		var raw [5]byte
		if _, err := r.Read(raw[:]); err != nil {
			return nil
		}
		quants[i] = parseRfxQuant(raw[:])
	}

	progQuants := make([]rfxProgQuant, numProgQuant)
	for i := 0; i < int(numProgQuant); i++ {
		var raw [16]byte
		if _, err := r.Read(raw[:]); err != nil {
			return nil
		}
		progQuants[i].quality = raw[0]
		progQuants[i].y = parseRfxQuant(raw[1:6])
		progQuants[i].cb = parseRfxQuant(raw[6:11])
		progQuants[i].cr = parseRfxQuant(raw[11:16])
	}

	// Compute the offset where the tile sub-blocks begin in `data`.
	tileOff, _ := r.Seek(0, 1)
	tileData := data[tileOff:]

	// Collect tile sub-blocks before dispatching so we can parallelise once
	// the count is high enough.  Same threshold as non-progressive rfx.go.
	type progTileWork struct {
		tileType uint16
		payload  []byte
	}
	var tiles []progTileWork
	offset := 0
	for offset+6 <= len(tileData) {
		tileType := leU16(tileData[offset:])
		tileLen := leU32(tileData[offset+2:])
		if tileLen < 6 || offset+int(tileLen) > len(tileData) {
			slog.Debug("RFX progressive: bad tile block length", "tileType", fmt.Sprintf("0x%04X", tileType), "tileLen", tileLen)
			break
		}
		switch tileType {
		case progWBTTileSimple, progWBTTileFirst, progWBTTileUpgrade:
			tiles = append(tiles, progTileWork{
				tileType: tileType,
				payload:  tileData[offset+6 : offset+int(tileLen)],
			})
		default:
			slog.Debug("RFX progressive: unknown tile type", "type", fmt.Sprintf("0x%04X", tileType))
		}
		offset += int(tileLen)
	}

	decodeOne := func(tw progTileWork, parallelComponents bool) {
		switch tw.tileType {
		case progWBTTileSimple:
			d.decodeTileSimple(tw.payload, quants, surface, surfData, outW, outH, parallelComponents, extrapolate)
		case progWBTTileFirst:
			d.decodeTileFirst(tw.payload, quants, progQuants, surface, surfData, outW, outH, parallelComponents, extrapolate)
		case progWBTTileUpgrade:
			d.decodeTileUpgrade(tw.payload, quants, progQuants, surface, surfData, outW, outH, ctxFlags, parallelComponents, extrapolate)
		}
	}

	const parallelTileThreshold = 12
	if len(tiles) >= parallelTileThreshold {
		// UPGRADE blocks must complete after the corresponding FIRST block has
		// run.  Within a single REGION block FreeRDP guarantees this by
		// ordering, but parallelising over a mix can shuffle that.  Split the
		// pool by tile type: do all FIRST/SIMPLE in parallel, then all UPGRADE
		// in parallel (real captures rarely mix them within one region).
		var firsts, upgrades []progTileWork
		for _, tw := range tiles {
			if tw.tileType == progWBTTileUpgrade {
				upgrades = append(upgrades, tw)
			} else {
				firsts = append(firsts, tw)
			}
		}
		runPhase := func(group []progTileWork) {
			if len(group) == 0 {
				return
			}
			workers := min(runtime.NumCPU(), len(group))
			ch := make(chan progTileWork, len(group))
			for _, tw := range group {
				ch <- tw
			}
			close(ch)
			var wg sync.WaitGroup
			for range workers {
				wg.Go(func() {
					defer func() {
						if rc := recover(); rc != nil {
							slog.Error("RFX progressive: tile decode panic", "err", rc)
						}
					}()
					for tw := range ch {
						decodeOne(tw, false)
					}
				})
			}
			wg.Wait()
		}
		runPhase(firsts)
		runPhase(upgrades)
	} else {
		for _, tw := range tiles {
			decodeOne(tw, true)
		}
	}

	return rects
}

// parseRfxQuant unpacks the 5-byte (10-nibble) RFX_COMPONENT_CODEC_QUANT
// layout.  Byte ordering matches FreeRDP's progressive.c line 59
// progressive_component_codec_quant_read.
func parseRfxQuant(data []byte) rfxQuant {
	return rfxQuant{
		LL3: data[0] & 0x0F,
		HL3: data[0] >> 4,
		LH3: data[1] & 0x0F,
		HH3: data[1] >> 4,
		HL2: data[2] & 0x0F,
		LH2: data[2] >> 4,
		HH2: data[3] & 0x0F,
		HL1: data[3] >> 4,
		LH1: data[4] & 0x0F,
		HH1: data[4] >> 4,
	}
}

func rfxGetQuant(quants []rfxQuant, idx int) rfxQuant {
	if idx >= 0 && idx < len(quants) {
		return quants[idx]
	}
	// Match FreeRDP's error path: an out-of-range quant index aborts the
	// tile, but we want to render *something* rather than crash, so fall
	// back to the lowest valid quant set.
	return rfxQuant{6, 6, 6, 6, 6, 6, 6, 6, 6, 6}
}

func safeSlice(data []byte, offset, length int) []byte {
	if length <= 0 || offset < 0 || offset+length > len(data) {
		return nil
	}
	return data[offset : offset+length]
}

// leU16/leU32 are tiny hot-path helpers; bytes.NewReader is too heavy for the
// inner block-walk inside parseRegion.
func leU16(b []byte) uint16 { return uint16(b[0]) | uint16(b[1])<<8 }
func leU32(b []byte) uint32 {
	return uint32(b[0]) | uint32(b[1])<<8 | uint32(b[2])<<16 | uint32(b[3])<<24
}
