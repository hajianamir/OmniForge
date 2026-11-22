# Audit Anchor Replication & Witness Guide

The optional `AuditCheckpointAnchor` (set `OMNIFORGE_AUDIT__ANCHOR_PATH`, keyed by
`OMNIFORGE_AUDIT__CHECKPOINT_KEY`) raises the bar for tampering with the audit log, but it is a
single-file, single-host construct. This guide states exactly what it does and does not prove,
and how to reach witness-grade guarantees with replication. If you skip to one section, read
[Honest Limits](#honest-limits).

## Threat model

| Attack | In-band chain MAC | + Anchor | + Replicated anchor / witness |
|---|---|---|---|
| Edit a field in a historical audit record | Detected (hash/MAC mismatch) | Detected | Detected |
| Re-chain the whole log with the default key | Detected (v2 has no unsigned path) | Detected (anchor MAC rejects the re-chain) | Detected |
| Rewrite the log but keep early records honest | Detected at the first edited link | Detected | Detected |
| **Truncate the tail of the log** (drop recent records) | **Not detected** | Detected (fewer records covered than last checkpoint) | Detected |
| **Truncate the tail of BOTH log and anchor consistently** | **Not detected** | **Not detected** | **Detected** |
| Compromise of `OMNIFORGE_AUDIT__CHECKPOINT_KEY` | n/a | **Anchor defeated** | **Partially mitigated** (replica limits how much can vanish unnoticed; witness makes rewriting visible) |
| Host lost / log destroyed | Availability loss | Recovery possible from replica | Recovery possible |

## Honest limits

1. **Consistent truncation is invisible without an external witness.** An attacker who can
   delete the trailing N audit records *and* the trailing N checkpoints leaves a perfectly
   self-consistent pair (shorter log, fewer checkpoints, all MACs valid). Nothing inside these
   two files can distinguish that from "those events never happened".
2. **The anchor key is the anchor.** Anyone holding `OMNIFORGE_AUDIT__CHECKPOINT_KEY` can mint
   fresh, valid checkpoints for a rewritten log. Keep it in a KMS/Vault, never on the same
   filesystem as the log, and rotate per the procedure below.
3. **An anchor proves existence and order, not content semantics.** It attests that N records
   with a given head hash existed at verification time; it does not prove who saw them.
4. **Cadence matters.** `OMNIFORGE_AUDIT__ANCHOR_EVERY=50` means up to 49 recent records are
   unanchored at any moment. Use `1` (every event) for high-value trails.
5. **Replication is only as independent as its target.** A copy on the same host/disk/volume is
   not a witness; an attacker with `root` takes both.

## Reference architectures

### Option A — Append-only replica on independent storage (baseline)

Ship the checkpoint file (and ideally the log) off-host continuously:

```bash
# /etc/cron.d/omniforge-anchor-replica — every minute
* * * * * omniforge  rsync -a --ignore-existing /var/lib/omniforge/audit_anchor.jsonl /mnt/worm_replica/audit_anchor/$(date +\%F).jsonl && logger -t omniforge-anchor "anchor replicated"
```

Prefer a destination that is *not writable by the application user*: an NFS/SMB share mounted
read-only for the API process, an SFTP drop, or object storage with versioning + Object Lock
(compliance mode retention ≥ your audit retention). Alert on two failure classes:

- **Sync lag** — replica's last line older than 5 minutes (SLO: under one anchor cadence).
- **Divergence** — running `AuditCheckpointAnchor.verify_against_log(log, replica_copy)` against
  the replica copy fails while the live copy passes, or vice versa.

### Option B — Collector-side verification (SIEM / syslog)

Forward both `audit.jsonl` and `audit_anchor.jsonl` to an independent collector (syslog,
OpenTelemetry, or a SIEM) that (1) stores them WORM, (2) runs
`AuditCheckpointAnchor.verify_against_log()` on a schedule, and (3) page on failure. Because the
collector holds its own copy, truncating the host copy is visible as "collector ahead of host".
This is usually the cheapest real witness for an existing SIEM-equipped org.

### Option C — External timestamp attestation (RFC 3161 / transparency witness)

Add an *external* anchor: after each checkpoint (or every M checkpoints), request an RFC 3161
timestamp token over the checkpoint line and store the token beside it, or cosign checkpoints
with a third-party witness (transparency-log style). An attacker rewriting log+anchor later
cannot backdate a cosignature, so even a full internal compromise becomes detectable for
records covered by an attestation. This is the only option that survives arbitrary host
compromise of the application server *and* its filesystems.

### Option D — Multi-node deployments

Each node must keep its **own** anchor file for its own log file (anchor files are not mergeable).
Do not point two writers at one anchor. Centralize verification on the collector (Option B):
N logs + N anchors, each verified independently. If you run one shared audit log behind several
writers (not the current default — `AuditLogger` is single-writer by design), you need Option C
or a serializer.

## Enabling the anchor (prerequisite)

```bash
OMNIFORGE_AUDIT__ANCHOR_PATH=/var/lib/omniforge/audit_anchor.jsonl   # separate file, ideally separate mount
OMNIFORGE_AUDIT__CHECKPOINT_KEY=<high-entropy secret from KMS>        # required in production
OMNIFORGE_AUDIT__ANCHOR_EVERY=1                                      # checkpoint every event
```

Behavioral contract (enforced, not advisory):

- A checkpoint write failure raises `SecurityError` — the triggering event is not considered logged.
- In `OMNIFORGE_ENVIRONMENT=production`, a missing/placeholder `CHECKPOINT_KEY` is a boot error.
- A checkpoint file that fails MAC verification or is not strictly seq-increasing is **refused at
  load** — the logger will not extend a forged anchor.
- Rollback (log shorter than anchor coverage) verifies but reports reduced coverage
  (`record N of M`) — alert on it; it means records after N are missing.

## Operations runbook

**Scheduled verification (hourly or after deploys):**

```bash
python - <<'PY'
from omniforge.core.security import AuditCheckpointAnchor
ok, checked, detail = AuditCheckpointAnchor.verify_against_log(
    "/var/lib/omniforge/audit.jsonl",
    "/var/lib/omniforge/audit_anchor.jsonl",
)
print(ok, checked, detail)
raise SystemExit(0 if ok else 2)
PY
```

Exit code 2 → page the security on-call. Do not "fix" by truncating files; snapshot both files
and the replica first.

**Routine rotation of `OMNIFORGE_AUDIT__CHECKPOINT_KEY`:**

The verifier cannot validate old checkpoints under a new key. Rotate by:
1. Verify the current anchor (`verify_with_anchor` via the app's `AuditLogger`, or the snippet
   above) and archive the old anchor as `audit_anchor.YYYY-MM-DD.jsonl`.
2. Set the new key; delete (or relocate) the stale anchor so it is not "extended" under the new key.
3. Keep the archived anchor + old key for offline verification of historical records.

**Incident response (verification failure or rollback coverage loss):**

1. Freeze writers: stop the API or set the audit path read-only.
2. Snapshot `audit.jsonl`, `audit_anchor.jsonl`, and the off-host replica (evidence).
3. Diff host vs replica; identify first divergent checkpoint seq (`verify_against_log` reports the
   checkpoint number and detail).
4. Restore from the replica if host files diverge; escalate if *both* diverge.
5. Rotate `OMNIFORGE_AUDIT__CHECKPOINT_KEY` and all secrets on the host before restart.

## Retention alignment

The anchor can only testify about records that still exist. If log retention deletes records
older than the anchor horizon (or vice versa), verification coverage silently shrinks. Keep:
**anchor retention ≥ audit log retention**, and document both in your data-retention policy.

## Summary

The anchor converts "the host can quietly rewrite history" into "history can only be rewritten
with the anchor key". Replication (Option A/B) converts that into "the host plus its entire
storage must be compromised". External attestation (Option C) converts it into "an external
party must be compromised too". Choose based on your threat model; do at least Option A.
