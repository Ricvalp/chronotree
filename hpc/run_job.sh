#!/usr/bin/env bash
# Signal-safe wrapper: a warning requests a coherent optimizer-boundary stop.
# A trapped USR1 interrupts bash wait; repeat wait and retain the CHILD exit code.
set -uo pipefail
if [[ $# -lt 2 ]]; then
  echo "usage: run_job.sh STOP_FILE COMMAND [ARG ...]" >&2
  exit 64
fi
STMEM_STOP_FILE=$1
shift
mkdir -p "$(dirname "$STMEM_STOP_FILE")"
# Allocation-specific filename makes removal safe and prevents stale-stop loops.
rm -f "$STMEM_STOP_FILE"
STMEM_SIGNAL_SEEN=0
stmem_request_stop() {
  STMEM_SIGNAL_SEEN=1
  : > "$STMEM_STOP_FILE"
}
trap stmem_request_stop USR1 TERM INT
"$@" &
STMEM_CHILD=$!
while true; do
  STMEM_SIGNAL_SEEN=0
  wait "$STMEM_CHILD"
  STMEM_EXIT=$?
  if [[ "$STMEM_SIGNAL_SEEN" -eq 1 ]]; then
    # The child may have exited concurrently; wait again obtains its real status.
    continue
  fi
  break
done
trap - USR1 TERM INT
exit "$STMEM_EXIT"
