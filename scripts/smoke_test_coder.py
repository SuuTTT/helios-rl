"""Smoke test: run as `coder` user to verify env + helios imports + write access."""
import os
import sys
import time
import csv

LOG_DIR = "/root/helios-rl/exp/smoke_test"
LOG_FILE = os.path.join(LOG_DIR, "smoke_result.csv")

def main():
    rows = []
    errors = []

    # 1. Basic user check
    user = os.environ.get("USER", os.popen("whoami").read().strip())
    rows.append(("user", user, "ok"))
    print(f"[1] Running as: {user}")

    # 2. Python version
    pyver = sys.version.split()[0]
    rows.append(("python_version", pyver, "ok"))
    print(f"[2] Python: {pyver}")

    # 3. JAX import + device check
    try:
        import jax
        import jax.numpy as jnp
        devices = jax.devices()
        device_str = str(devices[0])
        rows.append(("jax_device", device_str, "ok"))
        print(f"[3] JAX devices: {devices}")
    except Exception as e:
        errors.append(f"JAX: {e}")
        rows.append(("jax_device", "FAILED", str(e)))
        print(f"[3] JAX FAILED: {e}")

    # 4. Quick JAX compute (matmul)
    try:
        import jax.numpy as jnp
        a = jnp.ones((64, 64))
        result = float(jnp.sum(a @ a))
        rows.append(("jax_matmul_sum", str(result), "ok"))
        print(f"[4] JAX matmul sum: {result}")
    except Exception as e:
        errors.append(f"JAX compute: {e}")
        rows.append(("jax_matmul_sum", "FAILED", str(e)))

    # 5. helios import
    try:
        sys.path.insert(0, "/root/helios-rl/src")
        from helios.core.networks import ActorCritic
        rows.append(("helios_import", "ActorCritic", "ok"))
        print(f"[5] helios import OK")
    except Exception as e:
        errors.append(f"helios import: {e}")
        rows.append(("helios_import", "FAILED", str(e)))
        print(f"[5] helios import FAILED: {e}")

    # 6. Write CSV log to /root/helios-rl/exp/smoke_test/
    try:
        os.makedirs(LOG_DIR, exist_ok=True)
        with open(LOG_FILE, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["check", "value", "status"])
            writer.writerow(["timestamp", time.strftime("%Y-%m-%d %H:%M:%S"), "ok"])
            writer.writerows(rows)
        print(f"[6] Log written to {LOG_FILE}")
    except Exception as e:
        errors.append(f"write log: {e}")
        print(f"[6] Write FAILED: {e}")

    # Summary
    print("\n=== SMOKE TEST RESULT ===")
    if errors:
        print(f"FAILURES ({len(errors)}):")
        for e in errors:
            print(f"  - {e}")
        sys.exit(1)
    else:
        print(f"ALL CHECKS PASSED ({len(rows)} checks)")
        print(f"Log: {LOG_FILE}")

if __name__ == "__main__":
    main()
