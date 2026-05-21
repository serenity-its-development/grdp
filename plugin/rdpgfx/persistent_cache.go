package rdpgfx

import (
	"encoding/binary"
	"errors"
	"fmt"
	"log/slog"
	"os"
	"path/filepath"
)

// Persistent bitmap cache for MS-RDPEGFX.
//
// Windows hosts assume the client has retained a per-(cacheKey) on-disk
// copy of every SURFACE_TO_CACHE blit from prior sessions.  At session
// start they send a CACHE_IMPORT_OFFER listing the cacheKeys they expect
// us to already have; for any we acknowledge in the reply the server
// will skip re-sending those tiles (most prominently the lock-screen
// wallpaper).  Without a persistent store the server thinks we have
// content we don't and the affected regions render as solid black.
//
// File format (12-byte header + raw BGRA32 pixels):
//
//	magic     4 bytes  "GFXC"
//	version   1 byte   0x01
//	width     2 bytes  uint16 LE
//	height    2 bytes  uint16 LE
//	reserved  3 bytes  zero
//	pixels    width*height*4 bytes  BGRA32
//
// Total size = 12 + width*height*4.
const (
	gfxCacheHeaderSize = 12
	gfxCacheVersion    = 0x01
)

var gfxCacheMagic = [4]byte{'G', 'F', 'X', 'C'}

// errCacheFileBad is returned when an on-disk cache file fails magic,
// version, or size validation.  The caller treats it the same as
// "file does not exist" — skip the offered entry.
var errCacheFileBad = errors.New("rdpgfx: persistent cache file invalid")

// gfxCacheDir returns the directory used for persistent bitmap-cache
// files.  Honours RDPER_GFX_CACHE_DIR if set; otherwise falls back to
// os.UserCacheDir() + /rdper/gfxcache/.  The directory is NOT created
// here — callers that need it to exist (writers) call ensureGfxCacheDir.
func gfxCacheDir() (string, error) {
	if d := os.Getenv("RDPER_GFX_CACHE_DIR"); d != "" {
		return d, nil
	}
	base, err := os.UserCacheDir()
	if err != nil {
		return "", err
	}
	return filepath.Join(base, "rdper", "gfxcache"), nil
}

// ensureGfxCacheDir resolves the cache directory and creates it with
// mode 0700 if it does not already exist.
func ensureGfxCacheDir() (string, error) {
	dir, err := gfxCacheDir()
	if err != nil {
		return "", err
	}
	if err := os.MkdirAll(dir, 0o700); err != nil {
		return "", err
	}
	return dir, nil
}

// gfxCachePath returns the file path that backs a given cacheKey.
func gfxCachePath(dir string, key uint64) string {
	return filepath.Join(dir, fmt.Sprintf("%016x.bgra", key))
}

// writeGfxCacheEntry serialises a BGRA32 cache entry to disk atomically.
// pixels must be exactly width*height*4 bytes.  The on-disk file format
// is "GFXC" magic + version 0x01 + dimensions + pixel data; see the
// package-level comment for the byte layout.
func writeGfxCacheEntry(dir string, key uint64, width, height uint16, pixels []byte) error {
	if int(width)*int(height)*4 != len(pixels) {
		return fmt.Errorf("rdpgfx: pixel buffer size mismatch (%d vs %dx%dx4)",
			len(pixels), width, height)
	}
	buf := make([]byte, gfxCacheHeaderSize+len(pixels))
	copy(buf[0:4], gfxCacheMagic[:])
	buf[4] = gfxCacheVersion
	binary.LittleEndian.PutUint16(buf[5:7], width)
	binary.LittleEndian.PutUint16(buf[7:9], height)
	// buf[9:12] reserved bytes stay zero.
	copy(buf[gfxCacheHeaderSize:], pixels)
	return os.WriteFile(gfxCachePath(dir, key), buf, 0o600)
}

// readGfxCacheEntry loads a cache entry from disk and validates magic,
// version, and that the on-disk dimensions match the caller's expected
// width/height (taken from the CACHE_IMPORT_OFFER PDU).  Returns the
// pixel bytes on success, or an error otherwise.  Callers should treat
// any error as "skip this entry" — never panic, never propagate.
func readGfxCacheEntry(dir string, key uint64, wantW, wantH uint16) ([]byte, error) {
	path := gfxCachePath(dir, key)
	raw, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	if len(raw) < gfxCacheHeaderSize {
		return nil, errCacheFileBad
	}
	if raw[0] != gfxCacheMagic[0] || raw[1] != gfxCacheMagic[1] ||
		raw[2] != gfxCacheMagic[2] || raw[3] != gfxCacheMagic[3] {
		return nil, errCacheFileBad
	}
	if raw[4] != gfxCacheVersion {
		return nil, errCacheFileBad
	}
	w := binary.LittleEndian.Uint16(raw[5:7])
	h := binary.LittleEndian.Uint16(raw[7:9])
	if w != wantW || h != wantH {
		return nil, errCacheFileBad
	}
	expected := int(w) * int(h) * 4
	if len(raw)-gfxCacheHeaderSize != expected {
		return nil, errCacheFileBad
	}
	// Return a fresh slice (not aliasing into raw[12:]) so the caller can
	// hold onto it after raw is garbage-collected.
	pixels := make([]byte, expected)
	copy(pixels, raw[gfxCacheHeaderSize:])
	return pixels, nil
}

// deleteGfxCacheEntry removes the on-disk file for a cacheKey.  Missing
// files are not an error.
func deleteGfxCacheEntry(dir string, key uint64) {
	if err := os.Remove(gfxCachePath(dir, key)); err != nil && !errors.Is(err, os.ErrNotExist) {
		slog.Debug("RDPGFX: failed to delete cache file", "key", key, "err", err)
	}
}
