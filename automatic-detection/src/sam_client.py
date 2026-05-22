"""
Client for the SAM subprocess worker.

Usage
-----
    from sam_client import get_sam_client
    sam = get_sam_client()
    if sam.available:
        mask = sam.segment(image_rgb, point=(x, y))

The worker lives in a separate conda env (`sam_env`, numpy<2 + torch 2.2.2).
This client:
  * starts the worker on first use (singleton)
  * caches image embeddings by passing the same npy_path on repeated calls
  * writes temp numpy files for transport (fast + lossless)
  * falls back gracefully to None if the worker cannot start
"""
import os, sys, json, subprocess, threading, tempfile, atexit
import numpy as np

SAM_PYTHON = os.environ.get(
    "SAM_PYTHON",
    "/opt/anaconda3/envs/sam_env/bin/python",
)
WORKER_SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sam_worker.py")


class SamClient:
    def __init__(self):
        self.proc      = None
        self.available = False
        self.error     = None
        self._lock     = threading.Lock()
        self._tmpdir   = None

    # ── Lifecycle ────────────────────────────────────────────────────────────
    def start(self):
        if not os.path.exists(SAM_PYTHON):
            self.error = f"SAM python interpreter not found: {SAM_PYTHON}"
            return False
        if not os.path.exists(WORKER_SCRIPT):
            self.error = f"SAM worker script missing: {WORKER_SCRIPT}"
            return False

        try:
            self.proc = subprocess.Popen(
                [SAM_PYTHON, "-u", WORKER_SCRIPT],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
            )
        except Exception as e:
            self.error = f"failed to start worker: {e}"
            return False

        # Wait for the 'ready' line — worker emits it after loading the model.
        ready_line = self.proc.stdout.readline()
        if not ready_line:
            # Worker exited immediately — capture stderr for diagnostics
            stderr = self.proc.stderr.read() if self.proc.stderr else ""
            self.error = f"worker did not emit ready: {stderr.strip()[:400]}"
            return False

        try:
            status = json.loads(ready_line)
        except json.JSONDecodeError:
            self.error = f"bad ready line: {ready_line!r}"
            return False

        if not status.get("ok"):
            self.error = status.get("error", "unknown startup error")
            return False

        self._tmpdir = tempfile.mkdtemp(prefix="sam_")
        atexit.register(self.shutdown)
        self.available = True
        return True

    def shutdown(self):
        if self.proc and self.proc.poll() is None:
            try:
                self.proc.stdin.write(json.dumps({"cmd": "shutdown"}) + "\n")
                self.proc.stdin.flush()
                self.proc.wait(timeout=5)
            except Exception:
                self.proc.kill()
        self.proc = None
        self.available = False

    # ── Inference ────────────────────────────────────────────────────────────
    def segment(self, image_rgb, point=None, points=None, labels=None,
                box=None, multimask=True, select="highest_score",
                image_key=None):
        """
        Parameters
        ----------
        image_rgb : (H, W, 3) uint8  RGB image
        point     : (x, y)            single foreground point  (convenience)
        points    : list[(x, y)]      multiple prompt points
        labels    : list[int]         1=fg, 0=bg  (default all 1)
        box       : (x0, y0, x1, y1)  optional XYXY bounding box
        multimask : bool              return 3 candidate masks
        select    : str               'highest_score' or 'largest'
        image_key : str | None        reuse embedding if same key as previous call

        Returns
        -------
        mask : (H, W) bool  selected mask, or None on failure
        info : dict         scores, picked index, etc.
        """
        if not self.available:
            return None, {"error": "SAM not available"}

        if image_rgb.dtype != np.uint8:
            image_rgb = image_rgb.astype(np.uint8)
        if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
            return None, {"error": f"expected HxWx3, got {image_rgb.shape}"}

        # Assemble prompts
        if points is None and point is not None:
            points = [point]
        if points is None:
            points = []
        if labels is None:
            labels = [1] * len(points)

        # Write image and allocate mask paths (reuse if key provided)
        if image_key:
            npy_path = os.path.join(self._tmpdir, f"img_{image_key}.npy")
        else:
            npy_path = os.path.join(self._tmpdir, f"img_{os.getpid()}_{id(image_rgb)}.npy")

        if not os.path.exists(npy_path):
            np.save(npy_path, image_rgb)

        out_path = os.path.join(self._tmpdir, f"mask_{os.getpid()}_{id(self)}.npy")

        req = {
            "npy_path":  npy_path,
            "points":    [[float(p[0]), float(p[1])] for p in points],
            "labels":    [int(l) for l in labels],
            "out_path":  out_path,
            "multimask": bool(multimask),
            "select":    str(select),
        }
        if box is not None:
            req["box"] = [float(x) for x in box]

        with self._lock:
            try:
                self.proc.stdin.write(json.dumps(req) + "\n")
                self.proc.stdin.flush()
                resp_line = self.proc.stdout.readline()
            except (BrokenPipeError, OSError) as e:
                self.available = False
                return None, {"error": f"pipe broken: {e}"}

        if not resp_line:
            self.available = False
            return None, {"error": "worker closed stdout"}

        try:
            resp = json.loads(resp_line)
        except json.JSONDecodeError:
            return None, {"error": f"bad response: {resp_line!r}"}

        if not resp.get("ok"):
            return None, {"error": resp.get("error", "unknown")}

        mask = np.load(resp["out_path"])
        try:
            os.unlink(resp["out_path"])
        except OSError:
            pass

        return mask.astype(bool), {
            "scores": resp.get("scores", []),
            "picked": resp.get("picked"),
        }


# ── Module-level singleton ────────────────────────────────────────────────────
_CLIENT = None


def get_sam_client():
    """Return a singleton SamClient.  Starts the worker lazily on first call."""
    global _CLIENT
    if _CLIENT is None:
        _CLIENT = SamClient()
        _CLIENT.start()
    return _CLIENT
