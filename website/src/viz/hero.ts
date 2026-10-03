// The landing page's 3-D scene: no controls, one field, a slow idle orbit the
// visitor can take over. Loads the smallest useful set -- one field plus the
// region and clearance arrays, about 230 kB raw and 48 kB over the wire.

import { VolumeScene } from "./scene";
import { Payload } from "./volume";

export async function startHero(el: HTMLElement, baseUrl: string) {
  const p = new Payload(baseUrl);
  try {
    await p.load();
  } catch {
    el.innerHTML =
      '<p class="cap">The volume data is not available in this build.</p>';
    return;
  }
  const state = Object.keys(p.manifest.states)
    .find((s) => !s.startsWith("diff_")) ?? Object.keys(p.manifest.states)[0];
  const st = p.manifest.states[state];
  const key = st.fields["max_shear"] ?? Object.values(st.fields)[0];

  const scene = new VolumeScene(el, p);
  scene.setColormap("plasma");
  const [f, r] = await Promise.all([p.array(key), p.array(st.region)]);
  scene.setTextures(f, r);
  scene.start();

  // Idle rotation until the visitor touches it, then hand over for good.
  let t = 0, idle = true;
  const target = scene.target;
  const spin = () => {
    if (!idle) return;
    t += 0.0012;
    // 1.15 / 0.7: the figure 30 % smaller in frame, so the whole domain fits
    const R = p.extent[0] * 1.643;
    scene.camera.position.set(
      target.x + R * Math.cos(t), target.y + R * Math.sin(t),
      target.z + p.extent[2] * 1.93);
    scene.camera.lookAt(target);
    scene.render();
    requestAnimationFrame(spin);
  };
  el.addEventListener("pointerdown", () => { idle = false; }, { once: true });
  el.addEventListener("wheel", () => { idle = false; }, { once: true });
  requestAnimationFrame(spin);
  return scene;
}
