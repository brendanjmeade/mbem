// Three orthogonal slices through the volume, in a rotatable 3-D scene.
//
// Each slice is one quad whose fragment shader samples a 3-D texture at the
// fragment's own position in the box. That is why the threshold and the colour
// map are instant: they are uniforms, and nothing is recomputed on the CPU when
// a slider moves. Rebuilding a 2-D texture per slice would also have worked at
// this grid size, but it puts the controls on the wrong side of a 53x53 loop.
//
// Three textures, not one packed RGBA: `region` is a CATEGORICAL code and must
// use a NEAREST sampler, because linearly interpolating it invents regions that
// do not exist between the real ones. Filtering is per-sampler, not per-channel,
// so it cannot share a texture with the fields.
//
// uint8 everywhere and never FloatType: OES_texture_float_linear is at 54 % on
// iOS, and a float texture with LinearFilter there is texture-incomplete and
// samples as black with no error raised.

import {
  BufferGeometry, Color, Data3DTexture, DataTexture, DoubleSide,
  Float32BufferAttribute, Group, LineBasicMaterial, LineSegments, Mesh,
  NearestFilter, LinearFilter, PerspectiveCamera, PlaneGeometry, RedFormat,
  RGBAFormat, Scene, ShaderMaterial, UnsignedByteType, Vector3, WebGLRenderer,
  ClampToEdgeWrapping, GLSL3,
} from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { lut, type MapName } from "./colormaps";
import type { Payload } from "./volume";

const VERT = `
varying vec3 vBox;
uniform vec3 uOrigin, uExtent;
void main() {
  vec4 world = modelMatrix * vec4(position, 1.0);
  vBox = (world.xyz - uOrigin) / uExtent;        // [0,1]^3 in the sampled box
  gl_Position = projectionMatrix * viewMatrix * world;
}`;

// Declaring our own output is REQUIRED, not stylistic. three.js compiles every
// ShaderMaterial as "#version 300 es", but it injects
// `layout(location=0) out highp vec4 pc_fragColor` and the `gl_FragColor` alias
// only when glslVersion !== GLSL3 -- setting GLSL3 explicitly is the signal that
// the shader declares its own. Writing gl_FragColor here fails to compile with
// "cannot convert from 'highp 4-component vector of float' to 'const highp
// float'", which no type check or build step sees.
const FRAG = `
precision highp float;
precision highp sampler3D;
layout(location = 0) out vec4 fragColor;
varying vec3 vBox;
uniform sampler3D uField, uClear, uRegion;
uniform sampler2D uLut;
uniform float uThreshold;      // clearance_h, in the clearance array's units
uniform float uClearScale;     // level -> clearance_h
uniform bool  uShowOutside;
void main() {
  if (any(lessThan(vBox, vec3(0.0))) || any(greaterThan(vBox, vec3(1.0)))) discard;
  float reg = texture(uRegion, vBox).r * 255.0;
  if (reg < 0.5 && !uShowOutside) discard;       // not in the body at all
  float f = texture(uField, vBox).r;
  if (f <= 0.0) discard;                          // level 0 is "no data"
  float clear = texture(uClear, vBox).r * uClearScale;
  if (clear < uThreshold) discard;                // flagged, and hidden on ask
  vec3 c = texture(uLut, vec2(f, 0.5)).rgb;
  fragColor = vec4(c, 1.0);
}`;

function tex3d(data: Uint8Array, d: [number, number, number], smooth: boolean) {
  const t = new Data3DTexture(data, d[0], d[1], d[2]);
  t.format = RedFormat;
  t.type = UnsignedByteType;
  t.minFilter = t.magFilter = smooth ? LinearFilter : NearestFilter;
  t.wrapS = t.wrapT = t.wrapR = ClampToEdgeWrapping;
  t.unpackAlignment = 1;    // 53 and 103 are odd; the default 4 shears the rows
  t.needsUpdate = true;
  return t;
}

export interface SliceState { x: number; y: number; z: number; }

export class VolumeScene {
  readonly scene = new Scene();
  readonly camera: PerspectiveCamera;
  private renderer: WebGLRenderer;
  private controls: OrbitControls;
  private mat: ShaderMaterial;
  private planes: Mesh[] = [];
  private overlays = new Group();
  private p: Payload;
  private el: HTMLElement;
  private raf = 0;

  constructor(el: HTMLElement, p: Payload) {
    this.el = el;
    this.p = p;
    const [ex, ey, ez] = p.extent;
    const o = p.origin;

    this.scene.background = new Color(0xffffff);
    this.camera = new PerspectiveCamera(38, 1, 1, 1e4);
    // The model's vertical is +z (depth is negative z); three.js defaults to
    // +y up, which lays the box on its side and stands the free surface
    // vertically. OrbitControls reads this, so it must be set before them.
    this.camera.up.set(0, 0, 1);
    this.renderer = new WebGLRenderer({ antialias: true, alpha: false });
    this.renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
    el.appendChild(this.renderer.domElement);

    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.08;
    this.controls.target.set(o[0] + ex / 2, o[1] + ey / 2, o[2] + ez / 2);
    this.camera.position.set(o[0] + ex * 1.35, o[1] - ey * 1.05,
                            o[2] + ez * 1.9);
    // Orient the camera at the target NOW. OrbitControls only recomputes the
    // camera from its spherical state inside update(), and update() runs in
    // the rAF loop, which only starts on interaction -- so without this the
    // first frame is drawn with the camera's default orientation, pointing
    // away from the box, and the canvas is simply blank until you drag it.
    this.controls.update();

    const lutTex = new DataTexture(lut("viridis"), 256, 1, RGBAFormat);
    lutTex.minFilter = lutTex.magFilter = LinearFilter;
    lutTex.needsUpdate = true;

    this.mat = new ShaderMaterial({
      vertexShader: VERT, fragmentShader: FRAG, side: DoubleSide,
      glslVersion: GLSL3,
      uniforms: {
        uOrigin: { value: new Vector3(o[0], o[1], o[2]) },
        uExtent: { value: new Vector3(ex, ey, ez) },
        uField: { value: null }, uClear: { value: null },
        uRegion: { value: null }, uLut: { value: lutTex },
        uThreshold: { value: 0 }, uClearScale: { value: 1 },
        uShowOutside: { value: false },
      },
    });

    for (let axis = 0; axis < 3; axis++) {
      const g = new PlaneGeometry(1, 1);
      const m = new Mesh(g, this.mat);
      this.planes.push(m);
      this.scene.add(m);
    }
    this.scene.add(this.overlays);
    this.buildOverlays();
    this.setSlices({ x: 0.5, y: 0.5, z: 0.72 });
    this.resize();
    addEventListener("resize", () => this.resize());
  }

  /** Box edges, the fault outline and the inclusion rim, from geometry.json. */
  private buildOverlays() {
    const o = this.p.origin, [ex, ey, ez] = this.p.extent;
    const box = new LineSegments(
      edgesOf(o[0], o[1], o[2], ex, ey, ez),
      new LineBasicMaterial({ color: 0xbbbbbb }));
    this.overlays.add(box);

    const g = this.p.geometry;
    if (!g?.patches) return;
    for (const [name, b] of Object.entries<any>(g.patches)) {
      const isFault = b.is_fault;
      const inclusion = b.region === "inclusion" || name.startsWith("interface");
      if (!isFault && !inclusion) continue;
      const w = b.hi[0] - b.lo[0], h = b.hi[1] - b.lo[1], d = b.hi[2] - b.lo[2];
      const line = new LineSegments(
        edgesOf(b.lo[0], b.lo[1], b.lo[2], w, h, d),
        new LineBasicMaterial({ color: isFault ? 0xb5367a : 0x3b7ea1 }));
      this.overlays.add(line);
    }
    // The free surface, as a coarse wireframe from the height field, so the
    // hill is visible as geometry rather than only as a kink in the field.
    if (g.topography) this.overlays.add(topoLines(g.topography));
  }

  setTextures(field: Uint8Array, clear: Uint8Array, region: Uint8Array,
              clearMax: number) {
    const d = this.p.dims;
    const u = this.mat.uniforms;
    (u.uField.value as any)?.dispose?.();
    (u.uClear.value as any)?.dispose?.();
    (u.uRegion.value as any)?.dispose?.();
    u.uField.value = tex3d(field, d, true);
    u.uClear.value = tex3d(clear, d, true);
    u.uRegion.value = tex3d(region, d, false);   // NEAREST: categorical
    u.uClearScale.value = clearMax;
    this.render();
  }

  setColormap(name: MapName) {
    const t = new DataTexture(lut(name), 256, 1, RGBAFormat);
    t.minFilter = t.magFilter = LinearFilter;
    t.needsUpdate = true;
    (this.mat.uniforms.uLut.value as DataTexture).dispose();
    this.mat.uniforms.uLut.value = t;
    this.render();
  }

  setThreshold(v: number) {
    this.mat.uniforms.uThreshold.value = v;
    this.render();
  }

  /** Fractional slice positions in [0, 1]. */
  setSlices(s: SliceState) {
    const o = this.p.origin, [ex, ey, ez] = this.p.extent;
    const cx = o[0] + ex / 2, cy = o[1] + ey / 2, cz = o[2] + ez / 2;
    // yz plane at x
    this.planes[0].geometry.dispose();
    this.planes[0].geometry = new PlaneGeometry(ey, ez);
    this.planes[0].rotation.set(0, Math.PI / 2, Math.PI / 2);
    this.planes[0].position.set(o[0] + s.x * ex, cy, cz);
    // xz plane at y
    this.planes[1].geometry.dispose();
    this.planes[1].geometry = new PlaneGeometry(ex, ez);
    this.planes[1].rotation.set(Math.PI / 2, 0, 0);
    this.planes[1].position.set(cx, o[1] + s.y * ey, cz);
    // xy plane at z
    this.planes[2].geometry.dispose();
    this.planes[2].geometry = new PlaneGeometry(ex, ey);
    this.planes[2].rotation.set(0, 0, 0);
    this.planes[2].position.set(cx, cy, o[2] + s.z * ez);
    this.render();
  }

  setVisible(axis: 0 | 1 | 2, on: boolean) {
    this.planes[axis].visible = on;
    this.render();
  }

  resize() {
    const w = this.el.clientWidth || 640;
    const h = this.el.clientHeight || Math.round(w * 0.62);
    this.renderer.setSize(w, h, false);
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
    this.render();
  }

  /** Damped orbit needs a loop while the pointer is down; otherwise on demand. */
  start() {
    const tick = () => {
      this.raf = requestAnimationFrame(tick);
      if (this.controls.update()) this.renderer.render(this.scene, this.camera);
    };
    this.controls.addEventListener("start", () => { if (!this.raf) tick(); });
    this.controls.addEventListener("end", () => {
      cancelAnimationFrame(this.raf); this.raf = 0; this.render();
    });
    this.render();
  }

  /** The orbit centre, for a caller driving the camera itself. */
  get target(): Vector3 { return this.controls.target; }

  render() { this.renderer.render(this.scene, this.camera); }

  dispose() {
    cancelAnimationFrame(this.raf);
    this.controls.dispose();
    this.renderer.dispose();
  }
}

function edgesOf(x: number, y: number, z: number,
                 w: number, h: number, d: number): BufferGeometry {
  const c: number[][] = [];
  for (const i of [0, 1]) for (const j of [0, 1]) for (const k of [0, 1]) {
    c.push([x + i * w, y + j * h, z + k * d]);
  }
  const idx = [[0, 1], [0, 2], [0, 4], [1, 3], [1, 5], [2, 3], [2, 6],
               [3, 7], [4, 5], [4, 6], [5, 7], [6, 7]];
  const pos: number[] = [];
  for (const [a, b] of idx) pos.push(...c[a], ...c[b]);
  const g = new BufferGeometry();
  g.setAttribute("position", new Float32BufferAttribute(pos, 3));
  return g;
}

function topoLines(t: any): LineSegments {
  const { nx, ny, x0, y0, dx, dy, z } = t;
  const pos: number[] = [];
  const at = (i: number, j: number) => z[j * nx + i];
  const step = Math.max(1, Math.round(nx / 26));
  for (let j = 0; j < ny; j += step) {
    for (let i = 0; i + step < nx; i += step) {
      const a = at(i, j), b = at(i + step, j);
      if (a == null || b == null) continue;
      pos.push(x0 + i * dx, y0 + j * dy, a,
               x0 + (i + step) * dx, y0 + j * dy, b);
    }
  }
  for (let i = 0; i < nx; i += step) {
    for (let j = 0; j + step < ny; j += step) {
      const a = at(i, j), b = at(i, j + step);
      if (a == null || b == null) continue;
      pos.push(x0 + i * dx, y0 + j * dy, a,
               x0 + i * dx, y0 + (j + step) * dy, b);
    }
  }
  const g = new BufferGeometry();
  g.setAttribute("position", new Float32BufferAttribute(pos, 3));
  return new LineSegments(g, new LineBasicMaterial({ color: 0x999999 }));
}
