package rdpgfx

import (
	"bytes"
	"errors"
	"os"
	"path/filepath"
	"testing"
)

// makePixels builds a deterministic BGRA32 buffer of w*h pixels.  Each
// pixel encodes its (x,y) so any swapped or truncated read is obvious.
func makePixels(w, h int) []byte {
	out := make([]byte, w*h*4)
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			off := (y*w + x) * 4
			out[off+0] = byte(x)          // B
			out[off+1] = byte(y)          // G
			out[off+2] = byte((x + y) & 0xFF) // R
			out[off+3] = 0xFF             // A
		}
	}
	return out
}

// TestPersistentCacheRoundTrip writes a cache entry to disk and reads
// it back, verifying both pixel data and dimensions survive the
// serialise/deserialise cycle.
func TestPersistentCacheRoundTrip(t *testing.T) {
	dir := t.TempDir()
	const w, h uint16 = 16, 8
	const key uint64 = 0xCAFEBABE12345678
	pixels := makePixels(int(w), int(h))

	if err := writeGfxCacheEntry(dir, key, w, h, pixels); err != nil {
		t.Fatalf("writeGfxCacheEntry: %v", err)
	}
	got, err := readGfxCacheEntry(dir, key, w, h)
	if err != nil {
		t.Fatalf("readGfxCacheEntry: %v", err)
	}
	if !bytes.Equal(got, pixels) {
		t.Fatalf("pixel data round-trip mismatch: got %d bytes, want %d", len(got), len(pixels))
	}
}

// TestPersistentCacheRejectsBadMagic ensures readGfxCacheEntry returns
// an error (and does not panic) when a file with the wrong magic
// prefix exists at the expected path.
func TestPersistentCacheRejectsBadMagic(t *testing.T) {
	dir := t.TempDir()
	const key uint64 = 0xDEADBEEF
	// 12-byte garbage header + 4 bytes of "pixel" so total length is
	// the right size for 1x1; the magic bytes are wrong.
	bad := make([]byte, gfxCacheHeaderSize+4)
	copy(bad[0:4], []byte("XXXX"))
	bad[4] = gfxCacheVersion
	bad[5], bad[6] = 1, 0 // width = 1
	bad[7], bad[8] = 1, 0 // height = 1
	if err := os.WriteFile(gfxCachePath(dir, key), bad, 0o600); err != nil {
		t.Fatalf("seed bad file: %v", err)
	}
	_, err := readGfxCacheEntry(dir, key, 1, 1)
	if err == nil {
		t.Fatalf("expected error reading file with bad magic, got nil")
	}
	if !errors.Is(err, errCacheFileBad) {
		t.Fatalf("expected errCacheFileBad, got %v", err)
	}
}

// TestPersistentCacheRejectsDimensionMismatch ensures that an offer
// claiming different dimensions than the on-disk file causes the read
// to fail — the cache key is then treated as "not present" by the
// import-offer handler.
func TestPersistentCacheRejectsDimensionMismatch(t *testing.T) {
	dir := t.TempDir()
	const key uint64 = 0x1234567890ABCDEF
	const onDiskW, onDiskH uint16 = 32, 16
	pixels := makePixels(int(onDiskW), int(onDiskH))
	if err := writeGfxCacheEntry(dir, key, onDiskW, onDiskH, pixels); err != nil {
		t.Fatalf("writeGfxCacheEntry: %v", err)
	}
	// Offer claims 64x16 — does not match the on-disk 32x16.
	_, err := readGfxCacheEntry(dir, key, 64, onDiskH)
	if err == nil {
		t.Fatalf("expected error for dimension mismatch, got nil")
	}
	if !errors.Is(err, errCacheFileBad) {
		t.Fatalf("expected errCacheFileBad, got %v", err)
	}

	// Also flip the height to confirm both dimensions are checked.
	_, err = readGfxCacheEntry(dir, key, onDiskW, 999)
	if err == nil {
		t.Fatalf("expected error for height mismatch, got nil")
	}
}

// TestPersistentCacheMissingFile ensures a non-existent cache key
// returns os.ErrNotExist (or wraps it) so callers can distinguish
// "never cached" from "corrupted file".
func TestPersistentCacheMissingFile(t *testing.T) {
	dir := t.TempDir()
	_, err := readGfxCacheEntry(dir, 0xAAAA_BBBB_CCCC_DDDD, 1, 1)
	if err == nil {
		t.Fatalf("expected error for missing file, got nil")
	}
	if !errors.Is(err, os.ErrNotExist) {
		t.Fatalf("expected os.ErrNotExist, got %v", err)
	}
}

// TestPersistentCachePathFormat documents the on-disk filename
// convention so a future change to %016x cannot pass silently.
func TestPersistentCachePathFormat(t *testing.T) {
	dir := "/tmp/example"
	got := gfxCachePath(dir, 0x1)
	want := filepath.Join(dir, "0000000000000001.bgra")
	if got != want {
		t.Fatalf("gfxCachePath: got %q, want %q", got, want)
	}
}

// TestGfxCacheDirEnvOverride confirms RDPER_GFX_CACHE_DIR is honoured.
func TestGfxCacheDirEnvOverride(t *testing.T) {
	t.Setenv("RDPER_GFX_CACHE_DIR", "/tmp/override-gfxcache")
	dir, err := gfxCacheDir()
	if err != nil {
		t.Fatalf("gfxCacheDir: %v", err)
	}
	if dir != "/tmp/override-gfxcache" {
		t.Fatalf("gfxCacheDir: got %q, want override path", dir)
	}
}

// TestDeleteGfxCacheEntry confirms removal works and missing files are
// silent (no panic, no leaked error).
func TestDeleteGfxCacheEntry(t *testing.T) {
	dir := t.TempDir()
	const key uint64 = 0xFEEDFACE
	pixels := makePixels(4, 4)
	if err := writeGfxCacheEntry(dir, key, 4, 4, pixels); err != nil {
		t.Fatalf("write: %v", err)
	}
	deleteGfxCacheEntry(dir, key)
	if _, err := os.Stat(gfxCachePath(dir, key)); !errors.Is(err, os.ErrNotExist) {
		t.Fatalf("expected file gone, stat err = %v", err)
	}
	// Second delete should be a quiet no-op.
	deleteGfxCacheEntry(dir, key)
}
