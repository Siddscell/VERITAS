"""Quick live API smoke test against the running backend."""
import json
import os
import tempfile
import urllib.request

import numpy as np
from PIL import Image

# 1. healthz
resp = urllib.request.urlopen("http://127.0.0.1:8000/healthz")
print("healthz:", json.loads(resp.read()))

# 2. Create synthetic image
arr = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
img = Image.fromarray(arr)
tmp = tempfile.mktemp(suffix=".jpg")
img.save(tmp)

try:
    boundary = "----VeritasTestBoundary"
    with open(tmp, "rb") as f:
        img_bytes = f.read()

    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="test.jpg"\r\n'
        f"Content-Type: image/jpeg\r\n\r\n"
    ).encode() + img_bytes + (
        f"\r\n--{boundary}--\r\n"
    ).encode()

    req = urllib.request.Request(
        "http://127.0.0.1:8000/analyse",
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )
    with urllib.request.urlopen(req) as r:
        data = json.loads(r.read())

    print("\n=== /analyse response ===")
    print("verdict          :", data["verdict"])
    print("overall_confidence:", data["overall_confidence"])
    print("media_type       :", data["media_type"])

    s = next(a for a in data["agents"] if a.get("agent") == "Spatial Analysis")
    print("\nSpatial Analysis agent result:")
    for k, v in s.items():
        if k != "gradcam_b64":
            print(f"  {k}: {v!r}")
    print(f"  gradcam_b64: {s.get('gradcam_b64')}")
finally:
    if os.path.exists(tmp):
        os.remove(tmp)
