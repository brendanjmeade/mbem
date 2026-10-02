# Figure gallery

Empty by design. This is where `python -m mbem publish <run-dir>` copies the
figures you choose to keep, together with a line in `PROVENANCE.tsv` naming the
run, the commit and the file hash.

Nothing writes here as a side effect. Before this, every demo overwrote one
tracked image in place, so the committed figure was whatever ran last and nothing
recorded which code, which eps or which backend produced it -- the twelve images
this directory used to hold were outputs of scripts that no longer exist, with no
way to tell what they came from. They are in git history if anyone needs them.

A run keeps its own figures in its own directory under `runs/`; publishing is the
deliberate act of promoting one.
