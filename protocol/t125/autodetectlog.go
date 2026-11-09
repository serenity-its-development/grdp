package t125

// autodetectlog.go — what Windows' network test of this connection sees (a measurement, 2026-11-03).
//
// MS-RDPBCGR 2.2.14: the server times round trips (RTT requests) and bandwidth bursts (BW_START,
// BW_PAYLOADs, BW_STOP); the client answers with how long the burst took and how many bytes it
// received, and the server may tell the client what it concluded (RDP_NETCHAR_RESULT: base RTT,
// bandwidth in kbit/s, average RTT). Our answers report 0 bytes (mcs.go sendAutoDetectResponse);
// this logs what actually came, and every result Windows sends back. The answers are unchanged.

import (
	"bytes"
	"fmt"
	"log/slog"
	"sync"
	"time"

	"github.com/nakagami/grdp/core"
)

var adLog struct {
	sync.Mutex
	counts    map[uint16]int
	measuring bool
	start     time.Time
	payload   int // bytes the server declared in BW_PAYLOAD / BW_STOP payloads
	received  int // all MCS data bytes received during the burst
	lastBytes int // the last burst's received bytes, reported back (bwMeasuredBytes)
	lines     int
}

// bwMeasuredBytes is the number of bytes received during the bandwidth burst just stopped: the
// byteCount of RDP_BW_RESULTS (MS-RDPBCGR 2.2.14.2.2). It used to be reported as 0, and Windows
// then reckoned this loopback connection carried nothing and kept testing it (2026-11-03 17:40,
// ~50 bursts a second).
func bwMeasuredBytes() int {
	adLog.Lock()
	defer adLog.Unlock()
	return adLog.lastBytes
}

const adLogMaxLines = 200

func adInfo(msg string, args ...any) {
	if adLog.lines < adLogMaxLines {
		adLog.lines++
		slog.Info(msg, args...)
	}
}

// noteReceived counts an MCS data PDU of n bytes while a bandwidth burst is being timed.
func noteReceived(n int) {
	adLog.Lock()
	if adLog.measuring {
		adLog.received += n
	}
	adLog.Unlock()
}

// noteAutoDetect logs one auto-detect request; r is positioned after its requestType.
func noteAutoDetect(reqType uint16, r *bytes.Reader) {
	adLog.Lock()
	defer adLog.Unlock()
	if adLog.counts == nil {
		adLog.counts = map[uint16]int{}
	}
	adLog.counts[reqType]++
	n := adLog.counts[reqType]
	kind := fmt.Sprintf("%#04x", reqType)
	switch reqType {
	case rdpRttRequestConnecttime, rdpRttRequest:
		if n <= 3 || n%100 == 0 {
			adInfo("RDP autodetect: round-trip test", "type", kind, "count", n)
		}
	case rdpBwStartConnecttime, rdpBwStart, 0x0114:
		adLog.measuring, adLog.start, adLog.payload, adLog.received = true, time.Now(), 0, 0
	case rdpBwPayload:
		if l, err := core.ReadUint16LE(r); err == nil {
			adLog.payload += int(l)
		}
	case rdpBwStopConnecttime, rdpBwStop, 0x0629:
		if reqType == rdpBwStopConnecttime {
			if l, err := core.ReadUint16LE(r); err == nil {
				adLog.payload += int(l)
			}
		}
		ms := time.Since(adLog.start).Milliseconds()
		adLog.lastBytes = adLog.received
		if adLog.lastBytes < adLog.payload {
			adLog.lastBytes = adLog.payload
		}
		adInfo("RDP autodetect: bandwidth test", "type", kind, "count", n, "ms", ms,
			"payload_bytes", adLog.payload, "received_bytes", adLog.received, "we_report_bytes", adLog.lastBytes)
		adLog.measuring = false
	case 0x0840, 0x0880, 0x08C0: // RDP_NETCHAR_RESULT
		var base, bw, avg uint32
		switch reqType {
		case 0x0840:
			base, _ = core.ReadUInt32LE(r)
			avg, _ = core.ReadUInt32LE(r)
		case 0x0880:
			bw, _ = core.ReadUInt32LE(r)
			avg, _ = core.ReadUInt32LE(r)
		default:
			base, _ = core.ReadUInt32LE(r)
			bw, _ = core.ReadUInt32LE(r)
			avg, _ = core.ReadUInt32LE(r)
		}
		adInfo("RDP autodetect: Windows' result", "type", kind, "count", n,
			"base_rtt_ms", base, "bandwidth_kbps", bw, "average_rtt_ms", avg)
	default:
		if n <= 3 {
			adInfo("RDP autodetect: other request", "type", kind, "count", n)
		}
	}
}
