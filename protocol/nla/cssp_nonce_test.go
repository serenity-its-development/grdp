package nla

import (
	"bytes"
	"testing"
)

// TestEncodeDERTRequestNonce locks in the CredSSP v5+ ("updated") wire shape:
// Version must be 6 and the clientNonce (tag:5) must round-trip alongside the
// pubKeyAuth. A regression here is exactly what breaks NLA against patched /
// domain-joined Windows (Encryption Oracle Remediation).
func TestEncodeDERTRequestNonce(t *testing.T) {
	nonce := bytes.Repeat([]byte{0xAB}, 32)
	pk := []byte{1, 2, 3, 4, 5}

	req, err := DecodeDERTRequest(EncodeDERTRequest(nil, nil, pk, nonce))
	if err != nil {
		t.Fatalf("decode: %v", err)
	}
	if req.Version != 6 {
		t.Fatalf("Version = %d, want 6", req.Version)
	}
	if !bytes.Equal(req.ClientNonce, nonce) {
		t.Fatalf("ClientNonce = %x, want %x", req.ClientNonce, nonce)
	}
	if !bytes.Equal(req.PubKeyAuth, pk) {
		t.Fatalf("PubKeyAuth = %x, want %x", req.PubKeyAuth, pk)
	}

	// A negotiate-only request (no nonce) must omit tag:5 entirely.
	req2, err := DecodeDERTRequest(EncodeDERTRequest(nil, nil, nil, nil))
	if err != nil {
		t.Fatalf("decode negotiate: %v", err)
	}
	if len(req2.ClientNonce) != 0 {
		t.Fatalf("ClientNonce should be empty when not supplied, got %x", req2.ClientNonce)
	}
}
