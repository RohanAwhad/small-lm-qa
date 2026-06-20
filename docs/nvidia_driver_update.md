# NVIDIA Driver Update Workflow — H100 Cluster (CentOS Stream 9)

Tested on rh-h100-01 through rh-h100-09. June 2026.

## Prerequisites

- SSH access to the target node (`ssh rh-h100-XX`)
- `sudo` password for the `lab` user
- No active GPU workloads on the node
- Console/IPMI access as fallback (in case SSH breaks after reboot)

## Quick Reference

| Step | Command |
|------|---------|
| Check current state | `nvidia-smi \| head -5` |
| Remove runfile driver (if present) | `sudo /usr/bin/nvidia-uninstall --silent` |
| Remove RPM driver | `sudo dnf module remove --all nvidia-driver -y` |
| Reset module stream | `sudo dnf module reset nvidia-driver -y` |
| Remove old CUDA | `sudo dnf remove "cuda-*" "cudnn*" "libcudnn*" "libnccl*" "kmod-nvidia*" -y` |
| Refresh repo | `sudo dnf clean all && sudo dnf makecache` |
| Install prerequisites | `sudo dnf install kernel-devel-matched kernel-headers dkms gcc make -y` |
| Enable open-dkms | `sudo dnf module enable nvidia-driver:open-dkms -y` |
| Install driver | `sudo dnf install nvidia-open -y` |
| Install CUDA toolkit | `sudo dnf install cuda-toolkit nvidia-fabricmanager -y` |
| Build DKMS for new kernel | `sudo dkms build nvidia/VERSION -k $(uname -r)` |
| Enable services | `sudo systemctl enable nvidia-fabricmanager nvidia-persistenced` |
| Reboot | `sudo reboot` |
| Verify | `nvidia-smi` |

---

## Phase 0: Pre-Flight Checks

```bash
# Check current driver version
nvidia-smi | head -5

# Check what's installed (RPM vs runfile)
rpm -qa | grep -iE "nvidia-driver" | sort
cat /proc/driver/nvidia/version
ls /usr/bin/nvidia-uninstall  # exists = runfile install present

# Check for active GPU workloads
nvidia-smi --query-compute-apps=pid,name --format=csv

# Check who's logged in
who

# Check current kernel
uname -r
```

**Decision tree:**
- If `/usr/bin/nvidia-uninstall` exists → you have a runfile install, start at Phase 1
- If only RPM packages exist → skip to Phase 2
- If `rpm -qa | grep nvidia-driver` shows nothing → skip to Phase 3

---

## Phase 1: Remove Runfile-Installed Driver (if present)

Only needed if `/usr/bin/nvidia-uninstall` exists. This means someone installed the driver via a `.run` file, which conflicts with RPM management.

```bash
# Stop services
sudo systemctl stop nvidia-persistenced nvidia-fabricmanager

# Try to unload modules (may fail if something holds the GPU)
sudo rmmod nvidia_uvm nvidia_drm nvidia_modeset nvidia

# Run the uninstaller
sudo /usr/bin/nvidia-uninstall --silent

# Verify it's gone
ls /usr/bin/nvidia-uninstall  # should say "No such file"
```

If `rmmod` fails because modules are in use, check with `sudo fuser -v /dev/nvidia*` and kill the offending processes (nvtop, nvidia-smi daemons, etc). If you can't unload, the uninstaller will still work — the old modules get cleaned up on reboot.

---

## Phase 2: Remove Old RPM Driver Packages

```bash
# Remove driver module (removes driver RPMs + dependencies)
sudo dnf module remove --all nvidia-driver -y

# Reset the module stream selection
sudo dnf module reset nvidia-driver -y

# Remove old CUDA toolkit + libraries (avoids version conflicts)
sudo dnf remove "cuda-*" "cudnn*" "libcudnn*" "libnccl*" "kmod-nvidia*" -y

# Rebuild module dependency map
sudo depmod -a

# Verify clean slate
rpm -qa | grep -iE "nvidia-driver"  # should be empty
```

---

## Phase 3: Refresh Repo and Install Prerequisites

```bash
# Refresh DNF cache
sudo dnf clean all
sudo dnf makecache

# Check what driver version is available
dnf --disablerepo="*" --enablerepo="cuda*" list available nvidia-open

# Install kernel build tools + DKMS
sudo dnf install kernel-devel-matched kernel-headers dkms gcc make -y
```

**Important:** This step often installs a **new kernel**. Check:
```bash
rpm -q kernel-core | sort
```
If there's a newer kernel than what `uname -r` shows, the DKMS module needs to be built for it (see Phase 4b).

---

## Phase 4: Install Latest Driver + CUDA Toolkit

### 4a: Install packages

```bash
# Enable the open kernel module stream (recommended for H100)
sudo dnf module enable nvidia-driver:open-dkms -y

# Install driver (open kernel modules)
sudo dnf install nvidia-open -y

# Install CUDA toolkit + fabric manager
sudo dnf install cuda-toolkit nvidia-fabricmanager -y

# Enable services
sudo systemctl enable nvidia-fabricmanager nvidia-persistenced
```

### 4b: Build DKMS for new kernel (critical step!)

If a new kernel was installed in Phase 3, DKMS only auto-builds for the **currently running** kernel. You must manually build for the new kernel before rebooting:

```bash
# Check DKMS status — shows which kernels have the module
sudo dkms status

# Get the driver version from dkms status output (e.g. 610.43.02)
# Get the new kernel version from: rpm -q kernel-core | sort | tail -1

# Build and install for the new kernel
sudo dkms build nvidia/<VERSION> -k <NEW_KERNEL_VERSION>
sudo dkms install nvidia/<VERSION> -k <NEW_KERNEL_VERSION>

# Example:
# sudo dkms build nvidia/610.43.02 -k 5.14.0-710.el9.x86_64
# sudo dkms install nvidia/610.43.02 -k 5.14.0-710.el9.x86_64
```

---

## Phase 5: Reboot and Verify

```bash
sudo reboot
```

After reboot (wait ~2 minutes, then SSH back in):

```bash
# Verify driver loaded
nvidia-smi

# Expected output header:
# NVIDIA-SMI 610.43.02    KMD Version: 610.43.02    CUDA UMD Version: 13.3

# Verify all GPUs visible
nvidia-smi -L
# Should list all 8 H100 GPUs

# Verify kernel module
cat /proc/driver/nvidia/version

# Verify services running
systemctl is-active nvidia-persistenced nvidia-fabricmanager
```

---

## Future Updates (Day-2)

Once on a clean RPM install, future updates are simple:

```bash
# Same stream (patch update, e.g. 610.43 → 610.50):
sudo dnf update -y
# If kernel updated, build DKMS for new kernel (Phase 4b)
sudo reboot

# Different stream (major update, e.g. 610 → 620):
sudo dnf module reset nvidia-driver -y
sudo dnf module enable nvidia-driver:open-dkms -y
sudo dnf update --allowerasing -y
# Build DKMS for new kernel if needed
sudo reboot
```

---

## FAQ

### Q: How do I know if I have a runfile install vs RPM install?

Check for `/usr/bin/nvidia-uninstall` — if it exists, a runfile was used. Also compare `cat /proc/driver/nvidia/version` (loaded kernel module) against `rpm -qa | grep nvidia-driver` (RPM version). If they disagree, you have a mixed install.

### Q: Why open kernel modules instead of proprietary?

NVIDIA recommends open kernel modules for data center GPUs (H100, A100, etc) since driver 560+. They're the default path and get better support for Hopper/Blackwell architectures. Use `nvidia-driver:open-dkms` module stream.

### Q: Why DKMS instead of precompiled kmod?

The precompiled `kmod-nvidia` packages are tied to specific kernel versions. If CentOS Stream updates the kernel, the precompiled module breaks immediately. DKMS auto-rebuilds the module for new kernels (when booting into them after they're built). More resilient for rolling distros.

### Q: Do I need to rebuild the Python venv after updating?

No. The driver update doesn't affect Python packages. PyTorch/vLLM link against the CUDA runtime libraries, which are in the venv. They'll pick up the new driver automatically via the kernel module. However, if you also upgraded the CUDA toolkit major version (e.g. 12.x → 13.x), you may want to reinstall PyTorch to match.

### Q: What about nvidia-container-toolkit?

The container toolkit packages (`nvidia-container-toolkit`, `libnvidia-container`) are independent of the driver version. They should still work after a driver update. If containers can't see GPUs after update, run: `sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker`.

### Q: Can I update without a reboot?

Technically possible (unload old modules, load new ones) but unreliable — processes hold references to `/dev/nvidia*` and `rmmod` fails. Always reboot for driver updates.

---

## Troubleshooting

### nvidia-smi fails after reboot: "couldn't communicate with NVIDIA driver"

**Cause:** DKMS module wasn't built for the new kernel.

```bash
# Check which kernel you're running vs which DKMS modules exist
uname -r
sudo dkms status

# If the running kernel isn't listed, build it:
sudo dkms build nvidia/<VERSION> -k $(uname -r)
sudo dkms install nvidia/<VERSION> -k $(uname -r)
sudo modprobe nvidia
nvidia-smi  # should work now
```

### dnf install fails: "cccl-13-3 obsoletes cuda-cccl-12-9"

**Cause:** Old CUDA packages conflict with new ones.

```bash
# Remove all old CUDA packages first
sudo dnf remove "cuda-*" "cudnn*" "libcudnn*" "libnccl*" -y
# Then retry the install
sudo dnf install cuda-toolkit -y
```

### "Unable to find a match: nvidia-fabric-manager"

**Cause:** Package was renamed in newer releases.

```bash
# Search for the correct package name
dnf search nvidia-fabric
# Likely: nvidia-fabricmanager (no hyphen before "manager")
sudo dnf install nvidia-fabricmanager -y
```

### rmmod fails: "Module nvidia is in use"

**Cause:** Processes are holding GPU device files open.

```bash
# Find what's using the GPU
sudo fuser -v /dev/nvidia*

# Kill the processes (nvtop, nvidia-smi, training jobs, etc)
sudo kill <PIDs>

# Disable persistence mode
sudo nvidia-smi -pm 0

# Stop services
sudo systemctl stop nvidia-persistenced nvidia-fabricmanager

# Retry
sudo rmmod nvidia_uvm nvidia_drm nvidia_modeset nvidia
```

If all else fails, just proceed with the install and reboot — the old modules get replaced on reboot anyway.

### fabricmanager won't start: "PIDFile references legacy /var/run/"

**Cause:** Harmless systemd warning about old-style PID path. The service still works. Can be ignored.

### nvidia-smi shows wrong CUDA version after update

`nvidia-smi` shows the **driver's maximum supported CUDA version**, not the installed toolkit version. Check the actual toolkit with:
```bash
nvcc --version
# or
ls /usr/local/cuda/version.json
```

---

## Upgrade History

| Date | Node(s) | From | To |
|------|---------|------|----|
| 2026-06-09 | rh-h100-01 | 575.57.08 (runfile) + 550.90.07 (RPM) / CUDA 12.4-12.9 | 610.43.02 / CUDA 13.3 |
| 2026-06-09 | rh-h100-04 | 550.90.07 (RPM) / CUDA 12.4 | 610.43.02 / CUDA 13.3 |
