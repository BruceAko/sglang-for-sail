# SPDX-License-Identifier: Apache-2.0
"""End-to-end regression test for the DSA RAGGED top-k row-caliber fix.

What it guards
--------------
With an FP8 KV cache and the ``flashmla_sparse`` DSA prefill backend, the unfused
(fused top-k disabled) context-parallel prefill path used to fail on the very first
forward pass inside ``sglang.srt.layers.attention.dsa_backend.forward_extend``:
``topk_indices`` carries one row per padded/aligned query row while the RAGGED offset
is built per logical EXTEND token, so the masked addition raised a broadcast error
(or, after the fix, the row-caliber guards). This test starts a server with that
exact configuration, runs a short load and asserts that neither a traceback nor one
of those guards ever fires in the server log.

How to run
----------
Manual test (needs a checkpoint and four accelerators, so it is not CI-discovered).
From the repository root, on a machine with at least 4 accelerator devices of the
required capability and the model checkpoint available locally::

    SGLANG_TEST_MODEL_PATH=/path/to/glm-moe-mxfp4-checkpoint \\
    python3 test/manual/attention/test_dsa_fp8_kv_ragged_topk_rowcaliber_e2e.py -v

Environment variables (typed in ``sglang.srt.environ``; all read through ``.get()``)
-----------------------------------------------------------------------------------
``SGLANG_TEST_MODEL_PATH`` (``EnvStr``)
    Path to a GLM MoE MXFP4 checkpoint with FP8 per-channel dense layers (the model
    family that uses the DSA indexer). If it is unset or missing the test is skipped.
``SGLANG_TEST_PORT`` (``EnvInt``, default 31010)
    Port the spawned server listens on.
``SGLANG_TEST_SERVER_TIMEOUT`` (``EnvInt``, default 1800)
    Start-up budget in seconds.
``SGLANG_TEST_E2E_PROMPTS`` (``EnvInt``, default 4)
    Number of load requests sent by ``test_02``.
``SGLANG_TEST_E2E_INPUT_LEN`` (``EnvInt``, default 4096)
    Prompt length in tokens for those requests.
``SGLANG_TEST_DISABLE_MEGAMOE`` (``EnvBool``, default ``False``)
    Set to ``1``/``true`` to drop the Megamoe expert-path flags. The Megamoe per-rank
    token alignment is what pads query rows past the logical EXTEND count, so the
    default (Megamoe enabled) is the configuration that exercises the guard.

``SGLANG_DSA_FUSE_TOPK`` is forced to ``0`` by the test itself; export it to make the
intent explicit in the launch environment.

The test skips itself (with an explicit reason) when the checkpoint is unavailable,
fewer than four accelerator devices are visible, or their compute capability does not
match, so it is safe on CPU-only or single-accelerator hosts.
"""

import json
import os
import signal
import subprocess
import sys
import tempfile
import time
import unittest
import urllib.error
import urllib.request

from sglang.srt.environ import envs
from sglang.test.test_utils import CustomTestCase

REQUIRED_COMPUTE_CAPABILITY = (8, 9)
READY_MARKER = b"The server is fired up and ready to roll"
FORBIDDEN_MARKERS = (
    b"Traceback (most recent call last)",
    b"AssertionError",
    b"torch.AcceleratorError",
)


def _visible_device_count() -> int:
    try:
        import torch
    except Exception:  # torch is a hard dependency of the server, not of this helper
        return 0
    if not torch.cuda.is_available():
        return 0
    try:
        return int(torch.cuda.device_count())
    except Exception:
        return 0


def _capabilities() -> list:
    try:
        import torch
    except Exception:
        return []
    out = []
    for idx in range(_visible_device_count()):
        try:
            major, minor = torch.cuda.get_device_capability(idx)
            out.append((int(major), int(minor)))
        except Exception:
            continue
    return out


def _capability_ok() -> bool:
    caps = _capabilities()
    if not caps:
        return False
    return all(c >= REQUIRED_COMPUTE_CAPABILITY for c in caps)


class DsaFp8RaggedTopkRowCaliberTest(CustomTestCase):
    """Start the vulnerable configuration and prove the server stays healthy."""

    proc = None
    log_path = None
    log_dir = None
    log_file = None
    port = 0
    server_timeout = 0

    @classmethod
    def setUpClass(cls):
        model_path = envs.SGLANG_TEST_MODEL_PATH.get()
        if not model_path or not os.path.isdir(model_path):
            raise unittest.SkipTest(
                "SGLANG_TEST_MODEL_PATH must point to a local GLM MoE MXFP4 checkpoint"
            )
        if _visible_device_count() < 4:
            raise unittest.SkipTest(
                "this end-to-end test needs at least 4 visible accelerator devices"
            )
        if not _capability_ok():
            raise unittest.SkipTest(
                "this end-to-end test needs accelerator devices with compute "
                "capability >= %s" % (REQUIRED_COMPUTE_CAPABILITY,)
            )

        cls.port = envs.SGLANG_TEST_PORT.get()
        cls.server_timeout = envs.SGLANG_TEST_SERVER_TIMEOUT.get()
        cls.log_dir = tempfile.mkdtemp(prefix="dsa_fp8_ragged_topk_")
        cls.log_path = os.path.join(cls.log_dir, "server.log")

        # The defect only reproduces on the unfused DSA top-k transform.
        env = os.environ.copy()
        env["SGLANG_DSA_FUSE_TOPK"] = "0"
        env.pop("http_proxy", None)
        env.pop("https_proxy", None)
        env.pop("HTTP_PROXY", None)
        env.pop("HTTPS_PROXY", None)

        cmd = [
            sys.executable,
            "-m",
            "sglang.launch_server",
            "--model-path",
            model_path,
            "--host",
            "127.0.0.1",
            "--port",
            str(cls.port),
            "--tp-size",
            "4",
            "--ep-size",
            "4",
            "--attn-cp-size",
            "4",
            "--enable-prefill-cp",
            "--cp-strategy",
            "interleave",
            "--chunked-prefill-size",
            "32768",
            "--mem-fraction-static",
            "0.9",
            "--attention-backend",
            "dsa",
            "--dsa-prefill-backend",
            "flashmla_sparse",
            "--dsa-decode-backend",
            "flashmla_kv",
            "--kv-cache-dtype",
            "fp8_e4m3",
            "--disable-radix-cache",
            "--disable-cuda-graph",
            "--disable-overlap-schedule",
            "--max-running-requests",
            "16",
            "--num-continuous-decode-steps",
            "1",
            "--trust-remote-code",
            "--weight-loader-disable-mmap",
            "--watchdog-timeout",
            "24000",
            "--dist-timeout",
            "24000",
        ]
        # The reproducing configuration uses the Megamoe FP4 expert path: the token
        # alignment it imposes on each rank is what pads the query rows past the
        # logical EXTEND count. SGLANG_TEST_DISABLE_MEGAMOE drops those flags.
        if not envs.SGLANG_TEST_DISABLE_MEGAMOE.get():
            cmd += [
                "--moe-a2a-backend",
                "megamoe",
                "--enable-w4a4-mxfp4-megamoe",
                "--disable-shared-experts-fusion",
                "--disable-custom-all-reduce",
            ]
        cls.log_file = open(cls.log_path, "wb")
        cls.proc = subprocess.Popen(
            cmd,
            stdout=cls.log_file,
            stderr=subprocess.STDOUT,
            env=env,
            start_new_session=True,
        )

    @classmethod
    def tearDownClass(cls):
        proc = getattr(cls, "proc", None)
        if proc is not None and proc.poll() is None:
            for sig in (signal.SIGTERM, signal.SIGKILL):
                try:
                    os.killpg(os.getpgid(proc.pid), sig)
                except Exception:
                    try:
                        proc.terminate()
                    except Exception:
                        pass
                if proc.poll() is not None:
                    break
                time.sleep(15)
        log_file = getattr(cls, "log_file", None)
        if log_file is not None:
            try:
                log_file.close()
            except Exception:
                pass
        log_dir = getattr(cls, "log_dir", None)
        if log_dir:
            print("SERVER_LOG=%s" % os.path.join(log_dir, "server.log"))
            sys.stdout.flush()

    def _read_log(self) -> bytes:
        try:
            with open(self.log_path, "rb") as handle:
                return handle.read()
        except FileNotFoundError:
            return b""

    def _wait_until_ready(self):
        deadline = time.time() + self.server_timeout
        while time.time() < deadline:
            blob = self._read_log()
            for marker in FORBIDDEN_MARKERS:
                if marker in blob:
                    tail = blob.decode("utf-8", "replace")[-4000:]
                    self.fail(
                        "server log contains %r before readiness;\n%s" % (marker, tail)
                    )
            if READY_MARKER in blob:
                return
            if self.proc.poll() is not None:
                tail = self._read_log().decode("utf-8", "replace")[-4000:]
                self.fail(
                    "server exited with code %s before readiness:\n%s"
                    % (self.proc.returncode, tail)
                )
            time.sleep(5)
        self.fail("server not ready within %ss" % self.server_timeout)

    def _generate(self, input_ids, max_new_tokens=1):
        body = json.dumps(
            {
                "input_ids": input_ids,
                "sampling_params": {"max_new_tokens": max_new_tokens, "temperature": 0},
            }
        ).encode()
        req = urllib.request.Request(
            "http://127.0.0.1:%d/generate" % self.port,
            data=body,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=self.server_timeout) as handle:
            return json.loads(handle.read().decode())

    def test_01_server_reaches_ready_state(self):
        """The regression crashed at start-up warmup, so readiness alone is a gate."""
        self._wait_until_ready()

    def test_02_short_load_reports_no_error(self):
        self._wait_until_ready()
        num_prompts = envs.SGLANG_TEST_E2E_PROMPTS.get()
        input_len = envs.SGLANG_TEST_E2E_INPUT_LEN.get()
        failures = []
        for idx in range(num_prompts):
            # Deterministic pseudo-random prompt; the trigger depends on the padded
            # row layout, not on the prompt semantics.
            rnd = (idx * 7919 + 1013) % 90000
            input_ids = [rnd + ((j * 31) % 50000) for j in range(input_len)]
            try:
                self._generate(input_ids)
            except (urllib.error.URLError, TimeoutError) as exc:
                failures.append((idx, repr(exc)))
        self.assertEqual([], failures, "load requests failed: %s" % failures)

    def test_03_server_log_is_clean(self):
        self._wait_until_ready()
        blob = self._read_log()
        for marker in FORBIDDEN_MARKERS:
            self.assertNotIn(
                marker, blob, "forbidden marker %r found in server log" % marker
            )
        self.assertIn(
            "fp8_e4m3".encode(),
            blob,
            "server did not report the FP8 KV cache dtype; check the launch arguments",
        )
        self.assertIn(b"flashmla_sparse", blob)


if __name__ == "__main__":
    unittest.main(verbosity=2)
