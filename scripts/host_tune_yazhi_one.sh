#!/usr/bin/env bash
# ==============================================================================
# Yazhi-One — host tuning for long Adhan SLM CPU training runs
# ==============================================================================
# Everything here needs root, so it is kept separate from the launch script.
# Run once after a reboot (swap/swappiness persist via fstab/sysctl, the CPU
# governor does NOT survive a reboot unless re-applied).
#
#   sudo ./scripts/host_tune_yazhi_one.sh
#
# What it fixes, and why:
#
#   1. SWAP — the box had a single 4 GB swapfile at 100% utilisation with only
#      116 KiB free. That is the direct cause of the OOM kill that destroyed the
#      2026-10-27 run at step 99 (Out of memory: Killed process ... python,
#      anon-rss 3.5 GB). 16 GB of RAM is not enough to hold JAX + GNOME + a
#      browser; swap is what absorbs the spikes.
#
#   2. SWAPPINESS — the default 60 makes the kernel evict anonymous pages
#      aggressively. For this workload we want it to prefer dropping the
#      train.bin page cache (a file, cheap to re-read from NVMe) over swapping
#      out JAX's compiled executables (expensive to page back in). 10 is the
#      standard "server with a big working set" value.
#
#   3. CPU GOVERNOR — the 5600X was sitting in `powersave` at 99% of base clock.
#      A sustained matmul loop is exactly the case `performance` exists for.
#      Expect a small but free throughput gain; revert with
#      `sudo cpupower frequency-set -g powersave`.
# ==============================================================================

set -euo pipefail

if [[ "${EUID}" -ne 0 ]]; then
  echo "error: run this with sudo" >&2
  exit 1
fi

echo "==> 1. swap"
SWAPFILE=/swap2.img
SWAP_SIZE=16G

if swapon --show=NAME --noheadings | grep -qx "${SWAPFILE}"; then
  echo "    ${SWAPFILE} already active, skipping"
else
  if [[ ! -f "${SWAPFILE}" ]]; then
    echo "    creating ${SWAP_SIZE} swapfile at ${SWAPFILE}"
    # fallocate is instant, but some filesystems hand back holes that swapon
    # refuses; dd is the portable fallback.
    fallocate -l "${SWAP_SIZE}" "${SWAPFILE}" || \
      dd if=/dev/zero of="${SWAPFILE}" bs=1M count=16384 status=progress
    chmod 600 "${SWAPFILE}"
    mkswap "${SWAPFILE}"
  fi
  swapon "${SWAPFILE}"
  if ! grep -q "^${SWAPFILE}" /etc/fstab; then
    echo "${SWAPFILE} none swap sw 0 0" >> /etc/fstab
    echo "    added ${SWAPFILE} to /etc/fstab"
  fi
fi

echo
echo "==> 2. swappiness"
cat > /etc/sysctl.d/99-yazhi-training.conf <<'EOF'
# Yazhi-One training host: prefer dropping page cache (train.bin is mmap'd and
# cheap to re-read from NVMe) over swapping out JAX executables.
vm.swappiness = 10
# Keep a little page cache head-room so mmap reads do not stall on writeback.
vm.vfs_cache_pressure = 50
EOF
sysctl --system >/dev/null
echo "    vm.swappiness = $(cat /proc/sys/vm/swappiness)"

echo
echo "==> 3. CPU governor"
if command -v cpupower >/dev/null; then
  cpupower frequency-set -g performance >/dev/null
  echo "    governor = $(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor)"
else
  echo "    cpupower missing; writing sysfs directly"
  for g in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do
    echo performance > "${g}"
  done
  echo "    governor = $(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor)"
fi

echo
echo "==> done"
free -h
swapon --show
