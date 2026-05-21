package rdpgfx

// Progressive UPGRADE bitstream reader — the SRL/RAW dual-stream encoding
// used by PROGRESSIVE_WBT_TILE_UPGRADE blocks to refine the precision of a
// previously-decoded tile's coefficients.
//
// This is a faithful Go port of FreeRDP's progressive.c helpers
// progressive_rfx_srl_read (line 1075), progressive_rfx_upgrade_block (line
// 1203), and rawShift (line 1192).  The reader is MSB-first and pads with
// zeros past end-of-stream.
//
// Algorithm summary:
//
//   For each band, we have a "shift" left-shift count and a "numBits" extra
//   precision count.  If numBits == 0 the band is unchanged this pass.
//
//   For LL bands (state.nonLL == false): read numBits bits from the RAW
//   stream, treat as unsigned magnitude, shift << "shift", add to current[i].
//
//   For non-LL bands (state.nonLL == true): consult sign[i].  If sign[i] > 0
//   we know the coefficient is positive and we read numBits from RAW.  If
//   sign[i] < 0 we read numBits from RAW and negate.  If sign[i] == 0 the
//   coefficient was zero on the prior pass, so we don't yet know the sign;
//   read it from the SRL (sign-run-length) stream, which efficiently encodes
//   runs of zeros with a Golomb-Rice-derived code.

// progressiveBitStream is an MSB-first bit reader used for SRL/RAW streams.
type progressiveBitStream struct {
	data  []byte
	bytes int
	pos   int    // bit position
	acc   uint32 // 32-bit accumulator (MSB-aligned)
	mask  uint32 // FreeRDP keeps a "mask" field used as scratch; not strictly required here but kept for parity
}

func (s *progressiveBitStream) attach(b []byte) {
	s.data = b
	s.bytes = len(b)
	s.pos = 0
	s.acc = 0
	s.fetch()
}

// fetch loads up to 4 bytes into the high end of acc from the current byte
// position.  Pads with zeros if fewer than 4 bytes remain.  FreeRDP: macro
// BitStream_Fetch in winpr/bitstream.h.
func (s *progressiveBitStream) fetch() {
	bp := s.pos >> 3
	s.acc = 0
	for i := 0; i < 4 && bp+i < s.bytes; i++ {
		s.acc |= uint32(s.data[bp+i]) << uint(24-8*i)
	}
	// Discard any sub-byte residue from the position.  FreeRDP shifts the
	// accumulator left by (pos & 7); we mirror that.
	if shift := uint(s.pos & 7); shift != 0 {
		s.acc <<= shift
	}
}

// shift advances the cursor by n bits and refreshes the accumulator.
// FreeRDP: BitStream_Shift in winpr/bitstream.h.
func (s *progressiveBitStream) shift(n uint32) {
	if n == 0 {
		return
	}
	s.pos += int(n)
	if n < 32 {
		s.acc <<= n
	} else {
		s.acc = 0
	}
	// Re-fetch when we cross a byte boundary so the high end of acc always
	// reflects the bits at position (s.pos / 8).
	s.refill()
}

// refill recomputes acc from the byte pointer + current sub-byte offset.
// Done after every shift so the accumulator's MSB is always the next bit to
// read.  Equivalent to FreeRDP's combined BitStream_Shift+BitStream_Fetch.
func (s *progressiveBitStream) refill() {
	bp := s.pos >> 3
	sh := uint(s.pos & 7)
	s.acc = 0
	for i := 0; i < 4 && bp+i < s.bytes; i++ {
		s.acc |= uint32(s.data[bp+i]) << uint(24-8*i)
	}
	if sh != 0 {
		s.acc <<= sh
		// Also load one extra byte to fill the bits we shifted out.
		if bp+4 < s.bytes {
			s.acc |= uint32(s.data[bp+4]) >> (8 - sh)
		}
	}
}

// position returns the current bit position (used by upgrade_state_finish in
// FreeRDP; we don't presently expose that semantically but the field is here
// for parity).
func (s *progressiveBitStream) position() int { return s.pos }

// remaining returns the number of bits not yet consumed.
func (s *progressiveBitStream) remaining() int {
	tot := s.bytes * 8
	if s.pos >= tot {
		return 0
	}
	return tot - s.pos
}

// progressiveUpgradeState wraps the SRL+RAW bitstreams plus the SRL coder
// state (kp, nz, mode).  Mirrors RFX_PROGRESSIVE_UPGRADE_STATE in progressive.c
// at line 46.
type progressiveUpgradeState struct {
	srl, raw progressiveBitStream

	nonLL bool

	// SRL coder state.  FreeRDP: progressive.c lines 53-55.
	kp   uint32
	nz   int  // remaining zero-run count
	mode bool // false = zero-encoding, true = unary-encoding
}

func newProgressiveUpgradeState(srlData, rawData []byte) *progressiveUpgradeState {
	s := &progressiveUpgradeState{kp: 8}
	s.srl.attach(srlData)
	s.raw.attach(rawData)
	return s
}

// rawShift reads numBits from the RAW stream as an *unsigned* value.  FreeRDP:
// progressive.c rawShift at line 1192.
func (s *progressiveUpgradeState) rawShift(numBits uint32) int32 {
	if numBits == 0 {
		return 0
	}
	if numBits > 16 {
		numBits = 16
	}
	mask := (uint32(1) << numBits) - 1
	val := (s.raw.acc >> (32 - numBits)) & mask
	s.raw.shift(numBits)
	return int32(val)
}

// srlRead returns the next sign value for a zero-prior-pass coefficient,
// using FreeRDP's combined zero-encoding-then-unary-encoding scheme.  Returns
// −1, 0, or +1 (or a small-magnitude int when numBits > 1, used only by
// LL bands).  FreeRDP: progressive.c progressive_rfx_srl_read at line 1075.
func (s *progressiveUpgradeState) srlRead(numBits uint32) int32 {
	if s.nz > 0 {
		s.nz--
		return 0
	}

	k := s.kp / 8

	if !s.mode {
		// Zero-encoding phase.
		bit := (s.srl.acc >> 31) & 1
		s.srl.shift(1)

		if bit == 0 {
			// '0' bit: nz >= (1<<k), set nz = (1<<k), bump kp by 4.
			s.nz = int(uint32(1) << k)
			s.kp += 4
			if s.kp > 80 {
				s.kp = 80
			}
			s.nz--
			return 0
		}
		// '1' bit: nz < (1<<k), nz = next k bits, then switch to unary mode.
		s.nz = 0
		s.mode = true
		if k > 0 {
			mask := (uint32(1) << k) - 1
			s.nz = int((s.srl.acc >> (32 - k)) & mask)
			s.srl.shift(k)
		}
		if s.nz > 0 {
			s.nz--
			return 0
		}
	}

	// Unary-encoding phase: read sign bit, then magnitude (Golomb-Rice).
	s.mode = false
	sign := (s.srl.acc >> 31) & 1
	s.srl.shift(1)

	// Adjust kp downward by 6 (saturating at 0).
	if s.kp < 6 {
		s.kp = 0
	} else {
		s.kp -= 6
	}

	if numBits == 1 {
		if sign == 1 {
			return -1
		}
		return 1
	}

	// Magnitude: unary-coded count of 0-bits until a 1.  FreeRDP: progressive.c
	// lines 1145-1158.  Each 0-bit means "still go up to max"; the terminating
	// 1-bit ends the run.
	max := (uint32(1) << numBits) - 1
	mag := uint32(1)
	for mag < max {
		bit := (s.srl.acc >> 31) & 1
		s.srl.shift(1)
		if bit == 1 {
			break
		}
		mag++
	}
	if mag > 0x7FFF {
		mag = 0x7FFF
	}
	if sign == 1 {
		return -int32(mag)
	}
	return int32(mag)
}

// refineBlock applies one UPGRADE refinement pass to a single band.  buffer
// is the current coefficient buffer (post-shift); sign is the per-coefficient
// sign snapshot from the FIRST pass.  shift gives the new bits' left-shift
// count (per-band quant shift from progressive_rfx_upgrade_component); numBits
// gives how many extra precision bits per coefficient this pass contributes
// (0 ⇒ no-op).
//
// FreeRDP: progressive.c progressive_rfx_upgrade_block at line 1203.
func (s *progressiveUpgradeState) refineBlock(buffer, sign []int16, shift, numBits uint8) {
	if numBits == 0 {
		return
	}
	if numBits > 15 {
		// Defensive: malformed; treat as zero refinement.
		return
	}
	nb := uint32(numBits)
	sh := uint(shift)
	if sh > 15 {
		sh = 15
	}

	if !s.nonLL {
		// LL band: RAW only, no sign tracking.  FreeRDP: lines 1216-1227.
		for i := range buffer {
			input := s.rawShift(nb)
			shifted := input << sh
			buffer[i] = clampInt16(int32(buffer[i]) + shifted)
		}
		return
	}

	// Non-LL bands: consult per-coefficient sign tracking.  FreeRDP: lines
	// 1230-1253.
	for i := range buffer {
		var input int32
		switch {
		case sign[i] > 0:
			input = s.rawShift(nb)
		case sign[i] < 0:
			input = -s.rawShift(nb)
		default:
			input = s.srlRead(nb)
			sign[i] = clampInt16(input)
		}
		shifted := input << sh
		buffer[i] = clampInt16(int32(buffer[i]) + shifted)
	}
}

// clampInt16 saturates to the int16 range, matching FreeRDP's
// WINPR_ASSERTING_INT_CAST behaviour for INT16 (clamping in release builds).
func clampInt16(v int32) int16 {
	if v < -32768 {
		return -32768
	}
	if v > 32767 {
		return 32767
	}
	return int16(v)
}
