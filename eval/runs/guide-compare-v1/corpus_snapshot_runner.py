"""Run the scholar_lens CLI of the tree in the current directory with the
guide's gathered material fixed: the first run saves it, later runs (either
engine) reuse it, so both engines write from the same corpus."""

import os
import pickle
import runpy
import sys
from pathlib import Path

sys.path.insert(0, os.getcwd())
snapshot = Path(os.environ["GUIDE_CORPUS_SNAPSHOT"])

from scholar_lens.src import tech_guide as tg  # noqa: E402

assert tg.__file__.startswith(os.getcwd()), tg.__file__
original = tg.TechGuideGenerator.gather
STATE = (
    "_official",
    "_products",
    "_reference_pages",
    "_reference_ids",
    "_reference_searches",
)


async def gather(self, urls, **kwargs):
    if snapshot.exists():
        data = pickle.loads(snapshot.read_bytes())
        self._source_urls = tuple(urls)
        self._section_source_urls = set()
        for name, value in data["state"].items():
            setattr(self, name, value)
        self.notes.extend(data["notes"])
        print(f"corpus snapshot reused: {snapshot}", file=sys.stderr)
        return data["material"]
    material = await original(self, urls, **kwargs)
    state = {n: getattr(self, n) for n in STATE if hasattr(self, n)}
    snapshot.write_bytes(
        pickle.dumps({"material": material, "state": state, "notes": list(self.notes)})
    )
    print(f"corpus snapshot saved: {snapshot}", file=sys.stderr)
    return material


tg.TechGuideGenerator.gather = gather
sys.argv = ["scholar_lens", *sys.argv[1:]]
runpy.run_module("scholar_lens", run_name="__main__", alter_sys=True)
