"""Reference models shared by the gates and the studies.

These three builders are imported by 13 of the 20 msd gates and by 6 of the
demos, so they are library code, not examples -- they lived in ``examples/``
only because the gates reached in with a ``sys.path`` insert. As a package they
are a plain import from either side, which is what lets the last of that path
surgery go, and it makes "the gates and the demos build the SAME model" a
property of the import rather than a convention.

Kept verbatim from the scripts they came from, ``main()`` and argparse
included, so a case is still runnable:

    python -m mbem.cases.topo_inclusion --backend fmm
"""
