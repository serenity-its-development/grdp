package rdpgfx

import (
	"encoding/binary"
	"testing"
)

// TestProgressiveDecodeFraming exercises the top-level block walker with the
// infrastructure blocks that don't actually decode anything onto the surface:
// SYNC, FRAME_BEGIN/END, CONTEXT.  Empty input must return an empty slice
// without panic; a well-formed but tile-less stream must do the same.
func TestProgressiveDecodeFraming(t *testing.T) {
	d := newRfxProgressiveDecoder()
	surf := make([]byte, 64*64*4)

	if rects := d.Decode(nil, surf, 64, 64); rects != nil {
		t.Errorf("expected nil rects on nil input, got %d", len(rects))
	}

	// Well-formed SYNC (blockType=0xCCC0, blockLen=12, magic+version).
	var blk []byte
	put := func(b []byte) { blk = append(blk, b...) }

	bt := make([]byte, 6)
	binary.LittleEndian.PutUint16(bt[0:], progWBTSync)
	binary.LittleEndian.PutUint32(bt[2:], 12)
	put(bt)
	magic := make([]byte, 6)
	binary.LittleEndian.PutUint32(magic[0:], 0xCACCACCA)
	binary.LittleEndian.PutUint16(magic[4:], 0x0100)
	put(magic)

	// FRAME_BEGIN
	binary.LittleEndian.PutUint16(bt[0:], progWBTFrameBegin)
	binary.LittleEndian.PutUint32(bt[2:], 12)
	put(bt)
	body := make([]byte, 6)
	binary.LittleEndian.PutUint32(body[0:], 0) // frameIndex
	binary.LittleEndian.PutUint16(body[4:], 0) // regionCount
	put(body)

	// CONTEXT
	binary.LittleEndian.PutUint16(bt[0:], progWBTContext)
	binary.LittleEndian.PutUint32(bt[2:], 10)
	put(bt)
	ctx := make([]byte, 4)
	ctx[0] = 0  // ctxId
	binary.LittleEndian.PutUint16(ctx[1:], 64) // tileSize
	ctx[3] = rfxSubbandDiffing
	put(ctx)

	// FRAME_END
	binary.LittleEndian.PutUint16(bt[0:], progWBTFrameEnd)
	binary.LittleEndian.PutUint32(bt[2:], 6)
	put(bt)

	rects := d.Decode(blk, surf, 64, 64)
	if len(rects) != 0 {
		t.Errorf("expected no rects without any REGION block, got %d", len(rects))
	}

	// Truncated block — should return cleanly without panic.
	short := []byte{0xC0, 0xCC, 0x06, 0x00} // partial header
	if rects := d.Decode(short, surf, 64, 64); rects != nil {
		t.Errorf("expected nil rects for truncated input")
	}

	// Bad block length (< 6) — should bail out without panic.
	badLen := make([]byte, 6)
	binary.LittleEndian.PutUint16(badLen[0:], progWBTSync)
	binary.LittleEndian.PutUint32(badLen[2:], 2) // < 6
	if rects := d.Decode(badLen, surf, 64, 64); rects != nil {
		t.Errorf("expected nil rects for bad blockLen")
	}
}

// TestProgressiveDecodeTileSimpleEndToEnd builds a single TILE_SIMPLE block
// from a known coefficient vector, runs it through Decode, and asserts that
// at least one BGRA pixel got written to the destination surface.  We don't
// check the pixel value because the reverse transform of arbitrary
// coefficients is not a useful round-trip; we want to prove the code path
// runs to completion without panicking.
func TestProgressiveDecodeTileSimpleEndToEnd(t *testing.T) {
	d := newRfxProgressiveDecoder()
	surf := make([]byte, 64*64*4)

	// Coefficient vector: small magnitudes spread across all 4096 positions.
	coeffs := make([]int16, 4096)
	for i := range coeffs {
		coeffs[i] = int16(i % 7)
	}
	encoded := rlgr1Encode(coeffs)

	tileSimple := assembleTileSimple(0, 0, 0, 0, 0, encoded, encoded, encoded)
	region := assembleRegion([]byte{6, 6, 6, 6, 6}, tileSimple)
	stream := assembleStream(region)

	rects := d.Decode(stream, surf, 64, 64)
	if len(rects) == 0 {
		t.Fatalf("expected at least one rect from Decode")
	}

	// Surface must contain at least one non-zero pixel after decode.
	any := false
	for _, b := range surf {
		if b != 0 {
			any = true
			break
		}
	}
	if !any {
		t.Errorf("expected non-zero pixels on surface after TILE_SIMPLE decode")
	}
}

// TestProgressiveDecodeTileUpgradeReachesCache exercises an UPGRADE block
// targeting a tile that was previously seen via TILE_FIRST.  We verify that
// the tile state cache persists across separate Decode calls (UPGRADE in
// frame N+1 can reference FIRST from frame N) and that no panic occurs even
// when numBits is zero per band (a degenerate but legal upgrade).
func TestProgressiveDecodeTileUpgradeReachesCache(t *testing.T) {
	d := newRfxProgressiveDecoder()
	surf := make([]byte, 64*64*4)

	// FIRST pass.
	coeffs := make([]int16, 4096)
	for i := range coeffs {
		if i%19 == 0 {
			coeffs[i] = int16(i%30 - 15)
		}
	}
	encoded := rlgr1Encode(coeffs)
	tileFirst := assembleTileFirst(0, 0, 0, 0, 0, 0xFF, encoded, encoded, encoded)
	region1 := assembleRegion([]byte{6, 6, 6, 6, 6}, tileFirst)
	stream1 := assembleStream(region1)
	if rects := d.Decode(stream1, surf, 64, 64); len(rects) == 0 {
		t.Fatalf("FIRST decode returned no rects")
	}

	// Capture tile state to compare after UPGRADE.
	s := d.getSurface(surf, 64, 64)
	tileBefore := s.getTile(0, 0)
	if !tileBefore.allocated {
		t.Fatalf("expected tile state to be allocated after FIRST")
	}
	pass0 := tileBefore.pass

	// UPGRADE pass with zero data lengths (numBits delta will saturate to 0,
	// which makes refineBlock a no-op).  Still increments the pass counter.
	tileUpgrade := assembleTileUpgrade(0, 0, 0, 0, 0, 0xFF, nil, nil, nil, nil, nil, nil)
	region2 := assembleRegion([]byte{6, 6, 6, 6, 6}, tileUpgrade)
	stream2 := assembleStream(region2)
	d.Decode(stream2, surf, 64, 64)

	tileAfter := s.getTile(0, 0)
	if tileAfter.pass != pass0+1 {
		t.Errorf("expected pass to advance %d→%d, got %d", pass0, pass0+1, tileAfter.pass)
	}
}

// assembleStream wraps payloads in a SYNC/FRAME_BEGIN/CONTEXT/REGION/FRAME_END
// envelope so they get past the top-level dispatcher.
func assembleStream(region []byte) []byte {
	var out []byte
	push := func(blockType uint16, payload []byte) {
		bt := make([]byte, 6)
		binary.LittleEndian.PutUint16(bt[0:], blockType)
		binary.LittleEndian.PutUint32(bt[2:], uint32(6+len(payload)))
		out = append(out, bt...)
		out = append(out, payload...)
	}
	syncBody := make([]byte, 6)
	binary.LittleEndian.PutUint32(syncBody[0:], 0xCACCACCA)
	binary.LittleEndian.PutUint16(syncBody[4:], 0x0100)
	push(progWBTSync, syncBody)

	fbBody := make([]byte, 6)
	push(progWBTFrameBegin, fbBody)

	ctxBody := []byte{0, 64, 0, rfxSubbandDiffing}
	push(progWBTContext, ctxBody)

	push(progWBTRegion, region)
	push(progWBTFrameEnd, nil)
	return out
}

// assembleRegion builds a REGION payload with one tile of the given quant
// bytes and a single full-tile rect.
func assembleRegion(quant5 []byte, tileBlocks ...[]byte) []byte {
	tileData := []byte{}
	for _, t := range tileBlocks {
		tileData = append(tileData, t...)
	}
	out := []byte{
		64,                         // tileSize
		1, 0,                       // numRects = 1
		1,                          // numQuant = 1
		0,                          // numProgQuant = 0
		0,                          // flags (no extrapolate)
		1, 0,                       // numTiles = 1
		0, 0, 0, 0,                 // tileDataSize (filled in below)
		0, 0,  0, 0, 64, 0, 64, 0,  // rect: x=0 y=0 w=64 h=64
	}
	out = append(out, quant5...)
	binary.LittleEndian.PutUint32(out[8:], uint32(len(tileData)))
	out = append(out, tileData...)
	return out
}

// assembleTileSimple wraps Y/Cb/Cr blobs into a PROGRESSIVE_WBT_TILE_SIMPLE.
func assembleTileSimple(qY, qCb, qCr byte, xIdx, yIdx uint16, yData, cbData, crData []byte) []byte {
	payload := make([]byte, 16)
	payload[0] = qY
	payload[1] = qCb
	payload[2] = qCr
	binary.LittleEndian.PutUint16(payload[3:], xIdx)
	binary.LittleEndian.PutUint16(payload[5:], yIdx)
	payload[7] = 0
	binary.LittleEndian.PutUint16(payload[8:], uint16(len(yData)))
	binary.LittleEndian.PutUint16(payload[10:], uint16(len(cbData)))
	binary.LittleEndian.PutUint16(payload[12:], uint16(len(crData)))
	binary.LittleEndian.PutUint16(payload[14:], 0) // tailLen
	payload = append(payload, yData...)
	payload = append(payload, cbData...)
	payload = append(payload, crData...)
	return wrapBlock(progWBTTileSimple, payload)
}

func assembleTileFirst(qY, qCb, qCr byte, xIdx, yIdx uint16, quality byte, yData, cbData, crData []byte) []byte {
	payload := make([]byte, 17)
	payload[0] = qY
	payload[1] = qCb
	payload[2] = qCr
	binary.LittleEndian.PutUint16(payload[3:], xIdx)
	binary.LittleEndian.PutUint16(payload[5:], yIdx)
	payload[7] = 0
	payload[8] = quality
	binary.LittleEndian.PutUint16(payload[9:], uint16(len(yData)))
	binary.LittleEndian.PutUint16(payload[11:], uint16(len(cbData)))
	binary.LittleEndian.PutUint16(payload[13:], uint16(len(crData)))
	binary.LittleEndian.PutUint16(payload[15:], 0) // tailLen
	payload = append(payload, yData...)
	payload = append(payload, cbData...)
	payload = append(payload, crData...)
	return wrapBlock(progWBTTileFirst, payload)
}

func assembleTileUpgrade(qY, qCb, qCr byte, xIdx, yIdx uint16, quality byte, ySrl, yRaw, cbSrl, cbRaw, crSrl, crRaw []byte) []byte {
	payload := make([]byte, 20)
	payload[0] = qY
	payload[1] = qCb
	payload[2] = qCr
	binary.LittleEndian.PutUint16(payload[3:], xIdx)
	binary.LittleEndian.PutUint16(payload[5:], yIdx)
	payload[7] = quality
	binary.LittleEndian.PutUint16(payload[8:], uint16(len(ySrl)))
	binary.LittleEndian.PutUint16(payload[10:], uint16(len(yRaw)))
	binary.LittleEndian.PutUint16(payload[12:], uint16(len(cbSrl)))
	binary.LittleEndian.PutUint16(payload[14:], uint16(len(cbRaw)))
	binary.LittleEndian.PutUint16(payload[16:], uint16(len(crSrl)))
	binary.LittleEndian.PutUint16(payload[18:], uint16(len(crRaw)))
	payload = append(payload, ySrl...)
	payload = append(payload, yRaw...)
	payload = append(payload, cbSrl...)
	payload = append(payload, cbRaw...)
	payload = append(payload, crSrl...)
	payload = append(payload, crRaw...)
	return wrapBlock(progWBTTileUpgrade, payload)
}

func wrapBlock(blockType uint16, payload []byte) []byte {
	hdr := make([]byte, 6)
	binary.LittleEndian.PutUint16(hdr[0:], blockType)
	binary.LittleEndian.PutUint32(hdr[2:], uint32(6+len(payload)))
	return append(hdr, payload...)
}

// TestParseRegionAcceptsExtrapolate confirms we now process regions that carry
// the RFX_DWT_REDUCE_EXTRAPOLATE band-layout flag (previously skipped).  With
// no rects/tiles the region parses cleanly and yields zero rects (not a skip).
func TestParseRegionAcceptsExtrapolate(t *testing.T) {
	d := newRfxProgressiveDecoder()
	surf := make([]byte, 64*64*4)
	s := d.getSurface(surf, 64, 64)

	// 12-byte region header with flags bit 0 (extrapolate) set, no rects/quants.
	region := []byte{
		64,                      // tileSize
		0, 0,                    // numRects
		0,                       // numQuant
		0,                       // numProgQuant
		rfxDwtReduceExtrapolate, // flags
		0, 0,                    // numTiles
		0, 0, 0, 0,              // tileDataSize
	}
	rects := d.parseRegion(region, s, surf, 64, 64, 0)
	if len(rects) != 0 {
		t.Errorf("expected 0 rects for an empty extrapolate region, got %d", len(rects))
	}
}
