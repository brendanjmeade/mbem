# mbem website

Website for **mbem**, the mollified boundary element method described in

> Meade, B. J. (2026). Three dimensional non-singular mollified elastic
> dislocation theory for extended width fault zones and inhomogeneous boundary
> element models. *arXiv*:2609.15601. https://arxiv.org/abs/2609.15601

Built with [Astro](https://astro.build/). Pages:

- `/` — overview, with the stored solution in 3-D
- `/concepts/` — how the method works (content in `src/content/concepts.mdx`)
- `/examples/` — three slices through the solution, with the near-boundary
  quality threshold

## Development

```sh
npm install
npm run dev        # http://localhost:4321/mbem/
npm run build      # static site in dist/
npm run preview
```

The site is served under a base path (default `/mbem`, for a GitHub Pages project
site). Override with environment variables, e.g.
`BASE=/ SITE=https://example.org npm run build`. Fetches must go through
`import.meta.env.BASE_URL` — a hardcoded `/data/...` works in dev and 404s in
production.

## The data

Nothing is computed in the browser; the solve is Python. `public/data/` holds a
quantised readout of one stored run, regenerated with

```sh
python -m mbem sample runs/<study-dir> --spacing 8 --difference
python website/scripts/make_payload.py runs/<the-volume-dir>
```

The encoder is `mbem.webexport`, in the package rather than here, so it is
importable and gated (`tests/gates/mbem/verify_volume.py` clause [g]).

Two things about the format are load-bearing:

- **Fields are log10 in uint8.** They span about five decades, and linear
  quantisation is destructive rather than merely lossy — measured 99.7 % maximum
  relative error for linear uint8 and 39–54 % for linear uint16, against 2.4 %
  for log10 uint8.
- **Never float textures.** `OES_texture_float_linear` is unavailable on roughly
  half of iOS devices, and a float 3-D texture with linear filtering there is
  texture-incomplete: it samples as black, silently. uint8 is universal and 256
  levels exceed what a colour map can show.

Arrays are separate files fetched on demand, so first paint is ~230 kB raw
(~48 kB over the wire) out of ~2 MB total. `region` uses a NEAREST sampler and
cannot share a texture with the fields: it is a categorical code, and
interpolating it invents regions between the real ones.
