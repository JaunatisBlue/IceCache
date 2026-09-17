"""Build the optional native batch query against a local M-DCI checkout.

Run from IceCache/source with ICECACHE_MDCI_SOURCE pointing to the exact
source revision used for dciknn._dci.  This builds icecache._mdci_batch in
place and never overwrites the installed dciknn package.
"""

import os
import hashlib
import subprocess
from pathlib import Path

import numpy
from setuptools import Extension, setup


source = Path(os.environ["ICECACHE_MDCI_SOURCE"]).resolve()
if not (source / "include" / "dci.h").is_file():
    raise RuntimeError("ICECACHE_MDCI_SOURCE must name an M-DCI source checkout")
revision = subprocess.check_output(
    ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
).strip()
if revision != "1137dbbdad85abfb8c70c1f5b1c6fe30200bfc7e":
    raise RuntimeError("native batch requires the pinned M-DCI revision 1137dbbdad85")
if subprocess.check_output(
    ["git", "-C", str(source), "status", "--porcelain", "--", "src", "include"]
).strip():
    raise RuntimeError("M-DCI src/include must match the pinned clean checkout")
# Capsules do not expose a public ABI version.  Record the exact producer
# binary, and reject swapping it after building this extension.
import dciknn._dci as installed_dci
dci_sha = hashlib.sha256(Path(installed_dci.__file__).read_bytes()).hexdigest()
upstream_dci = (source / "src" / "dci.c").read_bytes()
if hashlib.sha256(upstream_dci).hexdigest() != (
    "60663f91c0de70b33c00f150726ba8c9b76efe236c84a025de504bbe6d52ee5f"
):
    raise RuntimeError("M-DCI source differs from tested revision 1137dbbdad85")
# Upstream allocates only 2*returned slots but writes the second channel at
# offset requested; a short result can otherwise write past the allocation.
old = b"nearest_neighbours[0] = (int*)malloc(sizeof(int) * cur_num_returned * 2);"
new = b"nearest_neighbours[0] = (int*)malloc(sizeof(int) * num_neighbours * 2);"
if upstream_dci.count(old) != 1:
    raise RuntimeError("expected M-DCI query allocation site not found")
patched = Path("build/mdci_batch_patched/dci.c")
patched.parent.mkdir(parents=True, exist_ok=True)
patched.write_bytes(upstream_dci.replace(old, new))
core = [
    "dci", "util", "debug", "hashtable_i", "hashtable_d", "btree_i",
    "btree_p", "hashtable_p", "hashtable_pp", "stack",
]
ext = Extension(
    "icecache._mdci_batch",
    sources=["icecache/mdci_batch.c"] + [str(patched) if x == "dci" else
             str(source / "src" / f"{x}.c") for x in core],
    include_dirs=[str(source / "include"), numpy.get_include()],
    define_macros=[("ICECACHE_DCI_SHA256", '"' + dci_sha + '"')],
    extra_compile_args=["-O3", "-std=gnu99", "-fopenmp", "-DUSE_OPENMP", "-march=core-avx2"],
    extra_link_args=["-fopenmp", "-lopenblas", "-lm"],
)
setup(name="icecache-mdci-batch", version="0.1.0", ext_modules=[ext])
