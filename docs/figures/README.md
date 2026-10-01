# Figure gallery

The **curated** copies of the study figures, one fixed filename per figure, and
the only figures tracked in git.

A study writes its figures beside itself in `studies/`, which is gitignored. It
does **not** touch this directory. Promoting a figure here is a deliberate act,
because the alternative is what this repo did before: every demo overwrote one
tracked PNG in place, so the committed image was simply the output of whoever
ran it last, with no record of which code or configuration produced it.

PDFs are ignored (regenerable at 500 dpi from the same script); the PNGs are
what `README.md` displays.
