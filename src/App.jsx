import React, { useEffect, useRef, useState, useCallback } from "react";
import * as THREE from "three";
import { GLTFLoader } from "three/examples/jsm/loaders/GLTFLoader.js";
import {
  Upload, RotateCw, RotateCcw, Printer, Save, Trash2, ChevronDown,
  Scissors, Sparkles, Layers, User, Wind, X, Check, RefreshCw,
} from "lucide-react";

const HEAD_MODEL_URL = "./models/head.glb";
/* Target vertical extent (in scene units) the scanned head model is
   normalized to, matching the footprint the rest of the scene (camera
   distance, panel width, grid) was tuned against. */
const HEAD_MODEL_TARGET_HEIGHT = 2.05;

/* ------------------------------------------------------------------ */
/* Constants / helpers                                                 */
/* ------------------------------------------------------------------ */

const SECTIONS = [
  { id: "nape", label: "ネープ", jp: "襟足", anchor: { theta: Math.PI * 0.92, phi: Math.PI * 0.02 } },
  { id: "middle", label: "ミドル", jp: "中間", anchor: { theta: Math.PI * 0.62, phi: Math.PI * 0.0 } },
  { id: "over", label: "オーバー", jp: "頭頂", anchor: { theta: Math.PI * 0.28, phi: Math.PI * 0.0 } },
  { id: "side", label: "サイド", jp: "側頭", anchor: { theta: Math.PI * 0.55, phi: Math.PI * 0.48 } },
];

const IRON_SIZES = [
  { id: "straight", label: "ストレート" },
  { id: "26", label: "26mm" },
  { id: "32", label: "32mm" },
];

const defaultSkull = { hachi: 0, roundness: 0, topHeight: 0 };
const defaultHairline = { napeHeight: 0, foreheadWidth: 0 };
const defaultHairFlow = {
  cowlickTheta: 0.08, cowlickPhi: -0.1, cowlickDir: "cw",
  napeGrowth: "down", frontGrowth: "left",
};
const defaultSections = () =>
  Object.fromEntries(
    SECTIONS.map((s) => [s.id, { elevation: 90, overDirection: 0, sliceTilt: 0, length: 14 }])
  );

const AI_STYLE_TEMPLATES = [
  { id: "a", label: "ショートレイヤー", napeLength: "刈り上げ気味", weight: "軽め", layerFeel: "強め",
    sections: { nape: { elevation: 110, overDirection: 0, sliceTilt: 5, length: 6 },
      middle: { elevation: 95, overDirection: 5, sliceTilt: 0, length: 9 },
      over: { elevation: 150, overDirection: 0, sliceTilt: -5, length: 10 },
      side: { elevation: 100, overDirection: -5, sliceTilt: 0, length: 8 } } },
  { id: "b", label: "ミディアムグラデーション", napeLength: "首にかかる長さ", weight: "重め", layerFeel: "控えめ",
    sections: { nape: { elevation: 60, overDirection: 0, sliceTilt: 0, length: 14 },
      middle: { elevation: 80, overDirection: 0, sliceTilt: 0, length: 17 },
      over: { elevation: 100, overDirection: 0, sliceTilt: 0, length: 19 },
      side: { elevation: 75, overDirection: 5, sliceTilt: 0, length: 16 } } },
  { id: "c", label: "ロングレイヤー", napeLength: "肩下", weight: "中間", layerFeel: "中間",
    sections: { nape: { elevation: 45, overDirection: 0, sliceTilt: 0, length: 24 },
      middle: { elevation: 70, overDirection: 0, sliceTilt: 0, length: 27 },
      over: { elevation: 130, overDirection: 0, sliceTilt: 0, length: 22 },
      side: { elevation: 65, overDirection: -5, sliceTilt: 0, length: 26 } } },
];

const clamp = (v, min, max) => Math.max(min, Math.min(max, v));
const sphToCart = (r, theta, phi) => new THREE.Vector3(
  r * Math.sin(theta) * Math.cos(phi),
  r * Math.cos(theta),
  r * Math.sin(theta) * Math.sin(phi)
);

const PANEL_W = 0.55;
const BAR_T = 0.02;

/* Shared scalp-surface scale function used by both the head mesh and any
   anchor point (section panels, cowlick) so nothing gets embedded inside
   the (non-spherical) deformed head. */
function computeSurfaceScale(n, skull, hairline) {
  const lat = Math.acos(clamp(n.y, -1, 1)); // 0 top .. PI bottom
  let scale = 1;

  const earBand = Math.exp(-Math.pow((lat - Math.PI * 0.5) / 0.35, 2));
  const sideWeight = Math.abs(n.x);
  scale += skull.hachi * 0.18 * earBand * sideWeight;

  if (n.z < -0.15) {
    const backWeight = Math.min(1, -n.z * 1.4);
    scale += skull.roundness * 0.18 * backWeight;
  }

  const topBand = Math.exp(-Math.pow(lat / 0.6, 2));
  scale += skull.topHeight * 0.22 * topBand;

  if (n.z < -0.1 && n.y < -0.1) {
    const napeBand = Math.exp(-Math.pow((lat - Math.PI * 0.72) / 0.3, 2));
    scale += hairline.napeHeight * 0.12 * napeBand;
  }
  if (n.z > 0.35 && n.y > -0.1) {
    const foreBand = Math.exp(-Math.pow((lat - Math.PI * 0.4) / 0.3, 2));
    scale += hairline.foreheadWidth * 0.1 * foreBand;
  }
  return scale;
}

/* The actual scalp surface point in a given spherical direction, including
   the vertical head-shape stretch — this is the true "root" a hair
   section should be anchored to. When a scanned head model (headData) is
   loaded, its real per-direction radius is used as the base instead of a
   unit sphere, so panels stay flush with the actual scalp. */
function headSurfacePoint(theta, phi, skull, hairline, headData) {
  const n = sphToCart(1, theta, phi).normalize();
  const scale = computeSurfaceScale(n, skull, hairline);
  if (headData) {
    const baseRadius = lookupHeadRadius(headData, n);
    return new THREE.Vector3(n.x * baseRadius * scale, n.y * baseRadius * scale, n.z * baseRadius * scale);
  }
  return new THREE.Vector3(n.x * scale, n.y * scale * 1.12, n.z * scale);
}

/* Build a deformed head geometry from skull + hairline params. Deforms the
   scanned head model (headData) when one is loaded, otherwise falls back
   to a procedural sphere. */
function buildHeadGeometry(skull, hairline, headData) {
  const geo = headData ? headData.geometry.clone() : new THREE.SphereGeometry(1, 48, 36);
  const pos = geo.attributes.position;
  const v = new THREE.Vector3();
  for (let i = 0; i < pos.count; i++) {
    v.fromBufferAttribute(pos, i);
    const r = v.length();
    if (headData && r < 1e-6) continue;
    const n = headData ? v.clone().divideScalar(r) : v.clone().normalize();
    const scale = computeSurfaceScale(n, skull, hairline);
    if (headData) {
      v.multiplyScalar(scale);
      pos.setXYZ(i, v.x, v.y, v.z);
    } else {
      v.multiplyScalar(scale);
      pos.setXYZ(i, v.x, v.y * 1.12, v.z);
    }
  }
  geo.computeVertexNormals();
  return geo;
}

/* Precompute a per-vertex (direction, radius) lookup table for a centered
   head mesh, used to approximate "what's the real scalp radius in this
   direction" without an expensive per-query raycast. */
function buildHeadLookup(geometry) {
  const pos = geometry.attributes.position;
  const count = pos.count;
  const dirs = new Float32Array(count * 3);
  const radii = new Float32Array(count);
  const v = new THREE.Vector3();
  for (let i = 0; i < count; i++) {
    v.fromBufferAttribute(pos, i);
    const r = v.length();
    radii[i] = r;
    if (r > 1e-6) {
      dirs[i * 3] = v.x / r;
      dirs[i * 3 + 1] = v.y / r;
      dirs[i * 3 + 2] = v.z / r;
    }
  }
  return { geometry, dirs, radii, count };
}

/* Weighted-nearest-direction radius lookup: finds the K vertices whose
   direction from center is closest to n and blends their radii, giving a
   smooth approximation of the scanned head's surface radius in direction n. */
function lookupHeadRadius(headData, n) {
  const { dirs, radii, count } = headData;
  const K = 12;
  const bestDot = new Float32Array(K).fill(-2);
  const bestIdx = new Int32Array(K).fill(-1);
  for (let i = 0; i < count; i++) {
    const dot = dirs[i * 3] * n.x + dirs[i * 3 + 1] * n.y + dirs[i * 3 + 2] * n.z;
    if (dot > bestDot[K - 1]) {
      let j = K - 1;
      while (j > 0 && bestDot[j - 1] < dot) {
        bestDot[j] = bestDot[j - 1];
        bestIdx[j] = bestIdx[j - 1];
        j--;
      }
      bestDot[j] = dot;
      bestIdx[j] = i;
    }
  }
  let wSum = 0, rSum = 0;
  for (let k = 0; k < K; k++) {
    const idx = bestIdx[k];
    if (idx < 0) continue;
    const w = Math.pow(Math.max(bestDot[k], 0), 8);
    wSum += w;
    rSum += w * radii[idx];
  }
  return wSum > 0 ? rSum / wSum : 1;
}

/* Center and uniformly scale the loaded glTF's first mesh to the scene's
   expected head footprint, then build the radius lookup table for it. */
function processScannedHead(gltf, onReady) {
  gltf.scene.updateMatrixWorld(true);
  let mesh = null;
  gltf.scene.traverse((child) => {
    if (!mesh && child.isMesh) mesh = child;
  });
  if (!mesh) return;
  const geo = mesh.geometry.clone();
  geo.applyMatrix4(mesh.matrixWorld);
  geo.computeBoundingBox();
  const box = geo.boundingBox;
  const center = box.getCenter(new THREE.Vector3());
  const size = box.getSize(new THREE.Vector3());
  geo.translate(-center.x, -center.y, -center.z);
  const scale = HEAD_MODEL_TARGET_HEIGHT / size.y;
  geo.scale(scale, scale, scale);
  geo.computeVertexNormals();
  onReady(buildHeadLookup(geo));
}

/* Load the scanned head model (GLB). Normally fetched from HEAD_MODEL_URL,
   but falls back to a base64 blob on window.__HEAD_MODEL_BASE64__ (parsed
   directly, no fetch) for static hosts that can't serve a .glb file by URL. */
function loadScannedHead(onReady) {
  const loader = new GLTFLoader();
  const onError = (err) => console.error("Failed to load scanned head model:", err);
  if (typeof window !== "undefined" && window.__HEAD_MODEL_BASE64__) {
    const binary = atob(window.__HEAD_MODEL_BASE64__);
    const bytes = new Uint8Array(binary.length);
    for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
    loader.parse(bytes.buffer, "", (gltf) => processScannedHead(gltf, onReady), onError);
    return;
  }
  loader.load(HEAD_MODEL_URL, (gltf) => processScannedHead(gltf, onReady), undefined, onError);
}

/* ------------------------------------------------------------------ */
/* Main component                                                      */
/* ------------------------------------------------------------------ */

export default function HairCutSimulator() {
  const mountRef = useRef(null);
  const threeRef = useRef({});
  const headDataRef = useRef(null);
  const [ready, setReady] = useState(false);
  const [headModelVersion, setHeadModelVersion] = useState(0);

  const [frontPhoto, setFrontPhoto] = useState(null);
  const [aiOptions, setAiOptions] = useState(null);
  const [selectedStyleId, setSelectedStyleId] = useState(null);
  const [freeDesign, setFreeDesign] = useState(false);

  const [viewMode, setViewMode] = useState("pullout"); // pullout | natural
  const [sliceMode, setSliceMode] = useState("vertical"); // vertical | horizontal
  const [ironSize, setIronSize] = useState("straight");
  const [curl, setCurl] = useState(100);

  const [skull, setSkull] = useState(defaultSkull);
  const [hairline, setHairline] = useState(defaultHairline);
  const [hairFlow, setHairFlow] = useState(defaultHairFlow);
  const [sections, setSections] = useState(defaultSections());
  const [activeSection, setActiveSection] = useState("nape");

  const [presets, setPresets] = useState([]);
  const [presetName, setPresetName] = useState("");
  const [rightTab, setRightTab] = useState("section"); // section | skull | flow

  const [printOpen, setPrintOpen] = useState(false);
  const [captures, setCaptures] = useState([]);
  const [notes, setNotes] = useState("");
  const [capturing, setCapturing] = useState(false);

  /* ---------------- load presets on mount ---------------- */
  useEffect(() => {
    try {
      const raw = localStorage.getItem("hair3d_presets");
      if (raw) setPresets(JSON.parse(raw));
    } catch (e) {}
  }, []);

  /* ---------------- three.js scene setup ---------------- */
  useEffect(() => {
    const mount = mountRef.current;
    if (!mount) return;
    const width = mount.clientWidth, height = mount.clientHeight;

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0xffffff);

    const camera = new THREE.PerspectiveCamera(42, width / height, 0.1, 100);
    const spherical = { radius: 4.6, theta: Math.PI / 2.15, phi: Math.PI / 2.3 };
    const target = new THREE.Vector3(0, 0, 0);

    const updateCamera = () => {
      const s = spherical;
      camera.position.set(
        target.x + s.radius * Math.sin(s.theta) * Math.cos(s.phi),
        target.y + s.radius * Math.cos(s.theta),
        target.z + s.radius * Math.sin(s.theta) * Math.sin(s.phi)
      );
      camera.lookAt(target);
    };
    updateCamera();

    const renderer = new THREE.WebGLRenderer({ antialias: true, preserveDrawingBuffer: true });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.setSize(width, height);
    mount.appendChild(renderer.domElement);

    const hemi = new THREE.HemisphereLight(0xffffff, 0xd7dde6, 1.15);
    scene.add(hemi);
    const key = new THREE.DirectionalLight(0xffffff, 0.55);
    key.position.set(3, 4, 2);
    scene.add(key);
    const fillLight = new THREE.DirectionalLight(0xffffff, 0.35);
    fillLight.position.set(-3, 1, -2);
    scene.add(fillLight);

    // white illustration-style head with a light contour wireframe overlay
    const headGroup = new THREE.Group();
    const headMat = new THREE.MeshStandardMaterial({ color: 0xffffff, roughness: 0.95, metalness: 0.0 });
    const headGeo = buildHeadGeometry(defaultSkull, defaultHairline);
    const headMesh = new THREE.Mesh(headGeo, headMat);
    headGroup.add(headMesh);
    const wireMat = new THREE.LineBasicMaterial({ color: 0xcbd5e1, transparent: true, opacity: 0.6 });
    const headWire = new THREE.LineSegments(new THREE.WireframeGeometry(headGeo), wireMat);
    headGroup.add(headWire);
    scene.add(headGroup);

    const grid = new THREE.GridHelper(6, 24, 0xe2e8f0, 0xf1f5f9);
    grid.position.y = -1.55;
    scene.add(grid);

    // cowlick marker
    const cowlickGeo = new THREE.ConeGeometry(0.045, 0.14, 12);
    const cowlickMat = new THREE.MeshStandardMaterial({ color: 0xe11d48, emissive: 0x4c0519 });
    const cowlickMesh = new THREE.Mesh(cowlickGeo, cowlickMat);
    scene.add(cowlickMesh);

    // section panels — textbook-style bold pink frame + pale pink fill + cut line + labels + direction arrow
    const panelGroup = new THREE.Group();
    scene.add(panelGroup);
    const panelObjs = {};
    SECTIONS.forEach((s) => {
      const grp = new THREE.Group();

      const fillGeo = new THREE.PlaneGeometry(PANEL_W, 1, 1, 1);
      fillGeo.translate(0, -0.5, 0);
      const fillMat = new THREE.MeshBasicMaterial({ color: 0xfbcfe8, transparent: true, opacity: 0.32, side: THREE.DoubleSide });
      const mesh = new THREE.Mesh(fillGeo, fillMat);
      grp.add(mesh);

      const barMat = new THREE.MeshBasicMaterial({ color: 0xe11d48 });
      const leftBar = new THREE.Mesh(new THREE.BoxGeometry(BAR_T, 1, BAR_T), barMat);
      const rightBar = new THREE.Mesh(new THREE.BoxGeometry(BAR_T, 1, BAR_T), barMat);
      const topBar = new THREE.Mesh(new THREE.BoxGeometry(PANEL_W + BAR_T, BAR_T, BAR_T), barMat);
      const bottomBar = new THREE.Mesh(new THREE.BoxGeometry(PANEL_W + BAR_T, BAR_T, BAR_T), barMat);
      leftBar.position.x = -PANEL_W / 2;
      rightBar.position.x = PANEL_W / 2;
      grp.add(leftBar, rightBar, topBar, bottomBar);

      const lineMat = new THREE.LineBasicMaterial({ color: 0x2563eb, linewidth: 2 });
      const lineGeo = new THREE.BufferGeometry().setFromPoints(
        Array.from({ length: 24 }, () => new THREE.Vector3())
      );
      const cutLine = new THREE.Line(lineGeo, lineMat);
      cutLine.renderOrder = 5;
      grp.add(cutLine);

      // curved direction-pull arrow guide (pink = forward, blue = backward)
      const arrowMat = new THREE.MeshBasicMaterial({ color: 0xf43f5e });
      const seedCurve = new THREE.QuadraticBezierCurve3(
        new THREE.Vector3(0, 0, 0), new THREE.Vector3(0, 0, 0.01), new THREE.Vector3(0, 0, 0.02)
      );
      const arrowTube = new THREE.Mesh(new THREE.TubeGeometry(seedCurve, 8, 0.009, 6, false), arrowMat);
      const arrowHead = new THREE.Mesh(new THREE.ConeGeometry(0.032, 0.09, 10), arrowMat);
      const arrowGroup = new THREE.Group();
      arrowGroup.add(arrowTube, arrowHead);
      arrowGroup.visible = false;
      grp.add(arrowGroup);

      panelGroup.add(grp);
      panelObjs[s.id] = {
        grp, mesh, cutLine, leftBar, rightBar, bottomBar,
        arrowGroup, arrowTube, arrowHead, arrowMat,
      };
    });

    threeRef.current = {
      scene, camera, renderer, spherical, target, updateCamera,
      headGroup, headMesh, headMat, headWire, cowlickMesh, panelObjs, mount,
    };

    let cancelled = false;
    loadScannedHead((headData) => {
      if (cancelled) return;
      headDataRef.current = headData;
      const t = threeRef.current;
      const newGeo = buildHeadGeometry(defaultSkull, defaultHairline, headData);
      t.headMesh.geometry.dispose();
      t.headMesh.geometry = newGeo;
      t.headWire.geometry.dispose();
      t.headWire.geometry = new THREE.WireframeGeometry(headData.geometry);
      setHeadModelVersion((v) => v + 1);
    });

    let raf;
    const loop = () => {
      renderer.render(scene, camera);
      raf = requestAnimationFrame(loop);
    };
    loop();

    // manual orbit controls (no OrbitControls available for this three version)
    let dragging = false, panning = false, lastX = 0, lastY = 0;
    const onDown = (e) => {
      dragging = true; panning = e.button === 2 || e.shiftKey;
      lastX = e.clientX; lastY = e.clientY;
    };
    const onUp = () => { dragging = false; panning = false; };
    const onMove = (e) => {
      if (!dragging) return;
      const dx = e.clientX - lastX, dy = e.clientY - lastY;
      lastX = e.clientX; lastY = e.clientY;
      if (panning) {
        const panSpeed = spherical.radius * 0.0016;
        const camRight = new THREE.Vector3();
        camera.getWorldDirection(camRight);
        const right = new THREE.Vector3().crossVectors(camera.up, camRight).normalize();
        target.addScaledVector(right, dx * panSpeed);
        target.y += dy * panSpeed;
      } else {
        spherical.phi -= dx * 0.006;
        spherical.theta = clamp(spherical.theta - dy * 0.006, 0.25, Math.PI - 0.25);
      }
      updateCamera();
    };
    const onWheel = (e) => {
      e.preventDefault();
      spherical.radius = clamp(spherical.radius + e.deltaY * 0.0025, 2.2, 9);
      updateCamera();
    };
    let touchLast = null, pinchDist = null;
    const onTouchStart = (e) => {
      if (e.touches.length === 1) { touchLast = [e.touches[0].clientX, e.touches[0].clientY]; }
      else if (e.touches.length === 2) {
        pinchDist = Math.hypot(
          e.touches[0].clientX - e.touches[1].clientX,
          e.touches[0].clientY - e.touches[1].clientY
        );
      }
    };
    const onTouchMove = (e) => {
      e.preventDefault();
      if (e.touches.length === 1 && touchLast) {
        const dx = e.touches[0].clientX - touchLast[0], dy = e.touches[0].clientY - touchLast[1];
        touchLast = [e.touches[0].clientX, e.touches[0].clientY];
        spherical.phi -= dx * 0.006;
        spherical.theta = clamp(spherical.theta - dy * 0.006, 0.25, Math.PI - 0.25);
        updateCamera();
      } else if (e.touches.length === 2 && pinchDist != null) {
        const d = Math.hypot(
          e.touches[0].clientX - e.touches[1].clientX,
          e.touches[0].clientY - e.touches[1].clientY
        );
        spherical.radius = clamp(spherical.radius - (d - pinchDist) * 0.01, 2.2, 9);
        pinchDist = d;
        updateCamera();
      }
    };
    const onTouchEnd = () => { touchLast = null; pinchDist = null; };

    const el = renderer.domElement;
    el.addEventListener("pointerdown", onDown);
    window.addEventListener("pointerup", onUp);
    window.addEventListener("pointermove", onMove);
    el.addEventListener("wheel", onWheel, { passive: false });
    el.addEventListener("contextmenu", (e) => e.preventDefault());
    el.addEventListener("touchstart", onTouchStart, { passive: true });
    el.addEventListener("touchmove", onTouchMove, { passive: false });
    el.addEventListener("touchend", onTouchEnd);

    const onResize = () => {
      const w = mount.clientWidth, h = mount.clientHeight;
      camera.aspect = w / h; camera.updateProjectionMatrix();
      renderer.setSize(w, h);
    };
    const ro = new ResizeObserver(onResize);
    ro.observe(mount);

    setReady(true);

    return () => {
      cancelled = true;
      cancelAnimationFrame(raf);
      ro.disconnect();
      el.removeEventListener("pointerdown", onDown);
      window.removeEventListener("pointerup", onUp);
      window.removeEventListener("pointermove", onMove);
      el.removeEventListener("wheel", onWheel);
      el.removeEventListener("touchstart", onTouchStart);
      el.removeEventListener("touchmove", onTouchMove);
      el.removeEventListener("touchend", onTouchEnd);
      mount.removeChild(renderer.domElement);
      renderer.dispose();
    };
  }, []);

  /* ---------------- update head geometry on skull/hairline change ---------------- */
  useEffect(() => {
    const t = threeRef.current;
    if (!t.headMesh) return;
    const headData = headDataRef.current;
    const newGeo = buildHeadGeometry(skull, hairline, headData);
    t.headMesh.geometry.dispose();
    t.headMesh.geometry = newGeo;
    // For the scanned head (dense mesh), the contour wireframe is rebuilt
    // once from the raw model on load rather than every slider tick, since
    // WireframeGeometry over ~70k triangles is too costly to redo on every
    // drag frame. The low-poly placeholder sphere can still afford it.
    if (t.headWire && !headData) {
      t.headWire.geometry.dispose();
      t.headWire.geometry = new THREE.WireframeGeometry(newGeo);
    }
  }, [skull, hairline, ready, headModelVersion]);

  /* ---------------- update cowlick marker ---------------- */
  useEffect(() => {
    const t = threeRef.current;
    if (!t.cowlickMesh) return;
    const theta = Math.PI * (0.06 + hairFlow.cowlickTheta);
    const phi = Math.PI * (0.5 + hairFlow.cowlickPhi) - Math.PI / 2;
    const surface = headSurfacePoint(theta, phi + Math.PI / 2, skull, hairline, headDataRef.current);
    const p = surface.clone().multiplyScalar(1.02);
    t.cowlickMesh.position.copy(p);
    t.cowlickMesh.lookAt(p.clone().multiplyScalar(2));
    t.cowlickMesh.rotateX(Math.PI / 2);
    t.cowlickMesh.material.color.set(hairFlow.cowlickDir === "cw" ? 0xe11d48 : 0x2563eb);
  }, [hairFlow, skull, hairline, ready, headModelVersion]);

  /* ---------------- update section panels + cutlines ---------------- */
  useEffect(() => {
    const t = threeRef.current;
    if (!t.panelObjs) return;
    const curlFactor = curl / 100;

    SECTIONS.forEach((s) => {
      const p = t.panelObjs[s.id];
      const sec = sections[s.id];
      const anchor = headSurfacePoint(s.anchor.theta, s.anchor.phi + Math.PI / 2, skull, hairline, headDataRef.current);
      p.grp.position.copy(anchor);

      // orientation: base outward normal (from scalp surface), tilt by elevation + overdirection
      const outward = anchor.clone().normalize();
      const elevRad = THREE.MathUtils.degToRad(sec.elevation);
      const overRad = THREE.MathUtils.degToRad(sec.overDirection);

      const natural = viewMode === "natural";
      const dir = new THREE.Vector3(0, -1, 0); // hanging straight down (natural fall)
      if (!natural) {
        // pull the panel outward/up according to elevation
        const up = new THREE.Vector3(0, 1, 0);
        const axis = new THREE.Vector3().crossVectors(outward, up).normalize();
        if (axis.lengthSq() < 1e-6) axis.set(1, 0, 0);
        dir.copy(outward).applyAxisAngle(axis, -(Math.PI / 2 - elevRad));
        dir.applyAxisAngle(outward, overRad);
      }
      p.grp.lookAt(p.grp.position.clone().add(dir));
      p.grp.rotateX(Math.PI / 2);

      const lengthUnits = Math.max(0.08, (sec.length / 15) * curlFactor);
      p.mesh.scale.set(1, lengthUnits, 1);

      // bold pink border frame tracks the panel's real rectangle (kept as
      // separate transforms, not the mesh's own scale, so line thickness
      // never distorts)
      p.leftBar.scale.y = lengthUnits; p.leftBar.position.y = -lengthUnits / 2;
      p.rightBar.scale.y = lengthUnits; p.rightBar.position.y = -lengthUnits / 2;
      p.bottomBar.position.y = -lengthUnits;

      // cutline geometry
      const pts = [];
      const tip = -lengthUnits;
      if (sliceMode === "vertical") {
        const tilt = THREE.MathUtils.degToRad(sec.sliceTilt);
        for (let i = 0; i < 24; i++) {
          const x = (i / 23 - 0.5) * PANEL_W;
          pts.push(new THREE.Vector3(x, tip + x * Math.tan(tilt) * 0.6, 0.004));
        }
      } else {
        // horizontal cross-check: curve follows head curvature at this depth
        const curveAmt = 0.09 * (1 + skull.roundness * 0.4);
        for (let i = 0; i < 24; i++) {
          const u = i / 23 - 0.5;
          const y = tip + Math.cos(u * Math.PI) * curveAmt - curveAmt;
          pts.push(new THREE.Vector3(u * PANEL_W, y, 0.004));
        }
      }
      p.cutLine.geometry.dispose();
      p.cutLine.geometry = new THREE.BufferGeometry().setFromPoints(pts);
      p.cutLine.material.color.set(sliceMode === "vertical" ? 0x2563eb : 0xdb2777);

      // curved direction-pull arrow guide: pink = forward direction, blue = backward
      const overAbs = Math.abs(sec.overDirection);
      if (overAbs > 3 && !natural) {
        p.arrowGroup.visible = true;
        const sign = Math.sign(sec.overDirection);
        const midY = -lengthUnits * 0.45;
        const start = new THREE.Vector3(PANEL_W * 0.5 + 0.03, midY + 0.07, 0.01);
        const ctrl = new THREE.Vector3(PANEL_W * 0.5 + 0.16 * sign, midY, 0.06);
        const end = new THREE.Vector3(PANEL_W * 0.5 + 0.05 * sign, midY - 0.09, 0.02);
        const curve = new THREE.QuadraticBezierCurve3(start, ctrl, end);
        p.arrowTube.geometry.dispose();
        p.arrowTube.geometry = new THREE.TubeGeometry(curve, 12, 0.009, 6, false);
        p.arrowHead.position.copy(end);
        const tangent = curve.getTangent(1).normalize();
        p.arrowHead.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), tangent);
        p.arrowMat.color.set(sign > 0 ? 0xf43f5e : 0x2563eb);
      } else {
        p.arrowGroup.visible = false;
      }
    });
  }, [sections, curl, sliceMode, viewMode, skull, hairline, ready, headModelVersion]);

  /* ---------------- photo upload -> mock AI prediction ---------------- */
  const onPhotoUpload = (e) => {
    const file = e.target.files?.[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = () => {
      setFrontPhoto(reader.result);
      setAiOptions(AI_STYLE_TEMPLATES);
      setSelectedStyleId(null);
    };
    reader.readAsDataURL(file);
  };

  const applyAiStyle = (style) => {
    setSelectedStyleId(style.id);
    setSections((prev) => ({ ...prev, ...style.sections }));
  };

  const resetAll = () => {
    setSkull(defaultSkull);
    setHairline(defaultHairline);
    setHairFlow(defaultHairFlow);
    setSections(defaultSections());
    setCurl(100);
    setAiOptions(null);
    setSelectedStyleId(null);
    setFrontPhoto(null);
  };

  /* ---------------- presets ---------------- */
  const savePreset = () => {
    if (!presetName.trim()) return;
    const preset = {
      id: Date.now(), name: presetName.trim(),
      data: { skull, hairline, hairFlow, sections, curl, ironSize, sliceMode, viewMode },
    };
    const next = [...presets, preset];
    setPresets(next);
    localStorage.setItem("hair3d_presets", JSON.stringify(next));
    setPresetName("");
  };
  const loadPreset = (preset) => {
    const d = preset.data;
    setSkull(d.skull); setHairline(d.hairline); setHairFlow(d.hairFlow);
    setSections(d.sections); setCurl(d.curl); setIronSize(d.ironSize);
    setSliceMode(d.sliceMode); setViewMode(d.viewMode);
  };
  const deletePreset = (id) => {
    const next = presets.filter((p) => p.id !== id);
    setPresets(next);
    localStorage.setItem("hair3d_presets", JSON.stringify(next));
  };

  /* ---------------- print / capture multi-angle ---------------- */
  const ANGLES = [
    { label: "斜め全景", theta: Math.PI / 2.15, phi: Math.PI / 2.3, radius: 4.6 },
    { label: "側面", theta: Math.PI / 2.05, phi: Math.PI, radius: 4.2 },
    { label: "後面", theta: Math.PI / 2.1, phi: -Math.PI / 2, radius: 4.2 },
    { label: "縦横クロスチェック", theta: Math.PI / 2.6, phi: Math.PI / 3, radius: 4.8 },
    { label: "自然落下シルエット", theta: Math.PI / 2.3, phi: Math.PI / 2.3, radius: 5.2, natural: true },
  ];

  const runCapture = useCallback(async () => {
    const t = threeRef.current;
    if (!t.renderer) return;
    setCapturing(true);
    const prevMode = viewMode;
    const shots = [];
    for (const a of ANGLES) {
      if (a.natural && viewMode !== "natural") setViewMode("natural");
      else if (!a.natural && viewMode === "natural") setViewMode("pullout");
      t.spherical.theta = a.theta; t.spherical.phi = a.phi; t.spherical.radius = a.radius;
      t.updateCamera();
      await new Promise((r) => requestAnimationFrame(() => requestAnimationFrame(r)));
      t.renderer.render(t.scene, t.camera);
      shots.push({ label: a.label, src: t.renderer.domElement.toDataURL("image/png") });
    }
    setViewMode(prevMode);
    setCaptures(shots);
    setCapturing(false);
    setPrintOpen(true);
  }, [viewMode]);

  const activeSec = sections[activeSection];
  const updateActiveSection = (patch) =>
    setSections((prev) => ({ ...prev, [activeSection]: { ...prev[activeSection], ...patch } }));

  /* ------------------------------------------------------------------ */
  return (
    <div className="min-h-screen bg-slate-950 text-slate-200 flex flex-col">
      {/* header */}
      <header className="flex items-center justify-between gap-3 px-4 py-3 border-b border-slate-800 bg-slate-950/95">
        <div className="flex items-center gap-2">
          <Scissors className="w-5 h-5 text-emerald-400" />
          <div>
            <h1 className="text-sm font-semibold text-slate-100 tracking-tight">3D カット展開図シミュレーター</h1>
            <p className="text-xs text-slate-500">骨格・毛流補正 / クロスチェック対応</p>
          </div>
        </div>
        <div className="flex items-center gap-2">
          <button
            onClick={() => setViewMode(viewMode === "pullout" ? "natural" : "pullout")}
            className="text-xs px-3 py-1.5 rounded-md border border-slate-700 bg-slate-900 hover:bg-slate-800 text-slate-300 flex items-center gap-1.5"
          >
            <Wind className="w-3.5 h-3.5" />
            {viewMode === "pullout" ? "引き出し展開図" : "自然落下"}
          </button>
          <button
            onClick={runCapture}
            disabled={capturing}
            className="text-xs px-3 py-1.5 rounded-md bg-indigo-600 hover:bg-indigo-500 text-white flex items-center gap-1.5 disabled:opacity-50"
          >
            <Printer className="w-3.5 h-3.5" />
            {capturing ? "撮影中…" : "印刷 / PDF出力"}
          </button>
        </div>
      </header>

      <div className="flex-1 grid grid-cols-1 lg:grid-cols-[280px_1fr_320px] gap-0">
        {/* LEFT: photo + AI + presets */}
        <aside className="border-b lg:border-b-0 lg:border-r border-slate-800 bg-slate-950/60 p-3 space-y-3 overflow-y-auto lg:max-h-[calc(100vh-57px)]">
          <details open className="group">
            <summary className="flex items-center justify-between cursor-pointer text-xs font-semibold text-slate-300 uppercase tracking-wide py-1">
              <span className="flex items-center gap-1.5"><Sparkles className="w-3.5 h-3.5 text-indigo-400" />スタイル入力 / AI予測</span>
              <ChevronDown className="w-3.5 h-3.5 text-slate-500 group-open:rotate-180 transition-transform" />
            </summary>
            <div className="pt-2 space-y-2">
              <label className="flex flex-col items-center justify-center gap-1.5 border border-dashed border-slate-700 rounded-lg p-4 text-xs text-slate-400 hover:border-indigo-500 cursor-pointer bg-slate-900/40">
                {frontPhoto ? (
                  <img src={frontPhoto} alt="正面写真" className="w-full h-28 object-cover rounded-md" />
                ) : (
                  <>
                    <Upload className="w-5 h-5 text-slate-500" />
                    正面写真をドラッグ＆ドロップ
                  </>
                )}
                <input type="file" accept="image/*" className="hidden" onChange={onPhotoUpload} />
              </label>

              {aiOptions && (
                <div className="space-y-1.5">
                  <p className="text-[11px] text-slate-500">AI予測パターンを選択（ネープ長・重さ・レイヤー感）</p>
                  {aiOptions.map((opt) => (
                    <button
                      key={opt.id}
                      onClick={() => applyAiStyle(opt)}
                      className={`w-full text-left rounded-md border px-2.5 py-2 text-xs transition-colors ${
                        selectedStyleId === opt.id
                          ? "border-emerald-500 bg-emerald-500/10 text-emerald-300"
                          : "border-slate-700 bg-slate-900/50 text-slate-300 hover:border-slate-600"
                      }`}
                    >
                      <div className="flex items-center justify-between">
                        <span className="font-medium">{opt.label}</span>
                        {selectedStyleId === opt.id && <Check className="w-3.5 h-3.5" />}
                      </div>
                      <div className="text-slate-500 mt-0.5">
                        {opt.napeLength} ・ 重さ:{opt.weight} ・ レイヤー:{opt.layerFeel}
                      </div>
                    </button>
                  ))}
                  <p className="text-[10px] text-slate-600 leading-relaxed">
                    ※ このデモ環境では実際の画像生成AIには接続していないため、3パターンの数値プリセットをAI予測の代わりに提示しています。選択後は右側パネルの手動スライダーで自由に上書きできます。
                  </p>
                </div>
              )}
            </div>
          </details>

          <div className="h-px bg-slate-800" />

          <div className="flex items-center justify-between">
            <span className="text-xs font-semibold text-slate-300 uppercase tracking-wide flex items-center gap-1.5">
              <User className="w-3.5 h-3.5 text-indigo-400" />フリーデザインモード
            </span>
            <button
              onClick={() => { setFreeDesign(!freeDesign); if (!freeDesign) resetAll(); }}
              className={`w-10 h-5 rounded-full relative transition-colors ${freeDesign ? "bg-indigo-600" : "bg-slate-700"}`}
            >
              <span className={`absolute top-0.5 w-4 h-4 rounded-full bg-white transition-all ${freeDesign ? "left-5" : "left-0.5"}`} />
            </button>
          </div>
          <p className="text-[10px] text-slate-600 leading-relaxed">
            写真を使わず、ゼロから右側の手動スライダーのみで展開図を組み上げます。
          </p>

          <div className="h-px bg-slate-800" />

          <details className="group">
            <summary className="flex items-center justify-between cursor-pointer text-xs font-semibold text-slate-300 uppercase tracking-wide py-1">
              <span className="flex items-center gap-1.5"><Save className="w-3.5 h-3.5 text-indigo-400" />プリセット保存</span>
              <ChevronDown className="w-3.5 h-3.5 text-slate-500 group-open:rotate-180 transition-transform" />
            </summary>
            <div className="pt-2 space-y-2">
              <div className="flex gap-1.5">
                <input
                  value={presetName}
                  onChange={(e) => setPresetName(e.target.value)}
                  placeholder="プリセット名"
                  className="flex-1 min-w-0 text-xs bg-slate-900 border border-slate-700 rounded-md px-2 py-1.5 text-slate-200 placeholder-slate-600 focus:outline-none focus:border-indigo-500"
                />
                <button onClick={savePreset} className="text-xs px-2.5 py-1.5 rounded-md bg-indigo-600 hover:bg-indigo-500 text-white shrink-0">保存</button>
              </div>
              <div className="space-y-1 max-h-40 overflow-y-auto">
                {presets.length === 0 && <p className="text-[11px] text-slate-600">保存済みプリセットはありません</p>}
                {presets.map((p) => (
                  <div key={p.id} className="flex items-center justify-between gap-1 text-xs bg-slate-900/50 border border-slate-800 rounded-md px-2 py-1.5">
                    <button onClick={() => loadPreset(p)} className="truncate text-left flex-1 text-slate-300 hover:text-emerald-300">{p.name}</button>
                    <button onClick={() => deletePreset(p.id)} className="text-slate-600 hover:text-red-400 shrink-0"><Trash2 className="w-3.5 h-3.5" /></button>
                  </div>
                ))}
              </div>
            </div>
          </details>
        </aside>

        {/* CENTER: 3D canvas */}
        <main className="relative flex flex-col min-h-[420px] lg:min-h-0">
          <div className="flex flex-wrap items-center gap-2 px-3 py-2 border-b border-slate-800 bg-slate-950/70">
            <div className="flex items-center rounded-md border border-slate-700 overflow-hidden">
              {[["vertical", "縦スライス"], ["horizontal", "横スライス(クロスチェック)"]].map(([id, label]) => (
                <button
                  key={id}
                  onClick={() => setSliceMode(id)}
                  className={`text-[11px] px-2.5 py-1.5 ${sliceMode === id ? "bg-emerald-600 text-white" : "bg-slate-900 text-slate-400 hover:bg-slate-800"}`}
                >
                  {label}
                </button>
              ))}
            </div>
            <div className="flex items-center rounded-md border border-slate-700 overflow-hidden">
              {IRON_SIZES.map((iz) => (
                <button
                  key={iz.id}
                  onClick={() => setIronSize(iz.id)}
                  className={`text-[11px] px-2.5 py-1.5 ${ironSize === iz.id ? "bg-indigo-600 text-white" : "bg-slate-900 text-slate-400 hover:bg-slate-800"}`}
                >
                  {iz.label}
                </button>
              ))}
            </div>
            <div className="flex items-center gap-2 ml-auto min-w-[180px]">
              <span className="text-[11px] text-slate-500 whitespace-nowrap">カール補正 {curl}%</span>
              <input
                type="range" min={100} max={140} value={curl}
                onChange={(e) => setCurl(Number(e.target.value))}
                className="w-28 accent-emerald-500"
              />
            </div>
          </div>

          <div ref={mountRef} className="flex-1 w-full min-h-[360px]" />

          <div className="flex items-center gap-3 px-3 py-1.5 border-t border-slate-800 bg-slate-950/70 text-[10px] text-slate-500">
            <span>ドラッグ: 回転</span><span>Shift+ドラッグ / 右ドラッグ: 平行移動</span><span>ホイール・ピンチ: ズーム</span>
          </div>
        </main>

        {/* RIGHT: precision controls */}
        <aside className="border-t lg:border-t-0 lg:border-l border-slate-800 bg-slate-950/60 p-3 space-y-3 overflow-y-auto lg:max-h-[calc(100vh-57px)]">
          <div className="flex rounded-md border border-slate-700 overflow-hidden text-[11px]">
            {[["section", "セクション"], ["skull", "骨格"], ["flow", "毛流・生え癖"]].map(([id, label]) => (
              <button
                key={id}
                onClick={() => setRightTab(id)}
                className={`flex-1 px-2 py-1.5 ${rightTab === id ? "bg-indigo-600 text-white" : "bg-slate-900 text-slate-400 hover:bg-slate-800"}`}
              >
                {label}
              </button>
            ))}
          </div>

          {rightTab === "section" && (
            <div className="space-y-3">
              <div className="flex flex-wrap gap-1.5">
                {SECTIONS.map((s) => (
                  <button
                    key={s.id}
                    onClick={() => setActiveSection(s.id)}
                    className={`text-[11px] px-2.5 py-1 rounded-full border ${
                      activeSection === s.id ? "border-emerald-500 bg-emerald-500/15 text-emerald-300" : "border-slate-700 text-slate-400 hover:border-slate-600"
                    }`}
                  >
                    {s.label}<span className="text-slate-600 ml-1">{s.jp}</span>
                  </button>
                ))}
              </div>

              <SliderRow label="引き出し角度 (Elevation)" unit="°" value={activeSec.elevation} min={0} max={180}
                onChange={(v) => updateActiveSection({ elevation: v })} />
              <SliderRow label="オーバーダイレクション" unit="°" value={activeSec.overDirection} min={-45} max={45}
                onChange={(v) => updateActiveSection({ overDirection: v })} />
              <SliderRow label="スライス線の傾斜" unit="°" value={activeSec.sliceTilt} min={-30} max={30}
                onChange={(v) => updateActiveSection({ sliceTilt: v })} help="前上がり ↔ 前下がり" />
              <SliderRow label="毛髪の長さ" unit="cm" value={activeSec.length} min={3} max={30}
                onChange={(v) => updateActiveSection({ length: v })} />
            </div>
          )}

          {rightTab === "skull" && (
            <div className="space-y-3">
              <p className="text-[11px] text-slate-500 flex items-center gap-1"><Layers className="w-3 h-3" />頭骨形状</p>
              <SliderRow label="ハチ張り幅" leftLabel="Narrow" rightLabel="Wide" value={skull.hachi} min={-1} max={1} step={0.05}
                onChange={(v) => setSkull((s) => ({ ...s, hachi: v }))} />
              <SliderRow label="絶壁・後頭部の丸み" leftLabel="Flat" rightLabel="Round" value={skull.roundness} min={-1} max={1} step={0.05}
                onChange={(v) => setSkull((s) => ({ ...s, roundness: v }))} />
              <SliderRow label="トップの高さ" leftLabel="Low" rightLabel="High" value={skull.topHeight} min={-1} max={1} step={0.05}
                onChange={(v) => setSkull((s) => ({ ...s, topHeight: v }))} />
              <div className="h-px bg-slate-800 my-2" />
              <p className="text-[11px] text-slate-500">生え際（ヘアライン）</p>
              <SliderRow label="ネープの高さ位置" leftLabel="High" rightLabel="Low" value={hairline.napeHeight} min={-1} max={1} step={0.05}
                onChange={(v) => setHairline((h) => ({ ...h, napeHeight: v }))} />
              <SliderRow label="前髪・おでこの広さ" leftLabel="Narrow" rightLabel="Wide" value={hairline.foreheadWidth} min={-1} max={1} step={0.05}
                onChange={(v) => setHairline((h) => ({ ...h, foreheadWidth: v }))} />
            </div>
          )}

          {rightTab === "flow" && (
            <div className="space-y-3">
              <p className="text-[11px] text-slate-500">つむじ（頭頂の渦）</p>
              <SliderRow label="位置（前後）" leftLabel="前寄り" rightLabel="後ろ寄り" value={hairFlow.cowlickTheta} min={-0.3} max={0.3} step={0.02}
                onChange={(v) => setHairFlow((h) => ({ ...h, cowlickTheta: v }))} />
              <SliderRow label="位置（左右）" leftLabel="左" rightLabel="右" value={hairFlow.cowlickPhi} min={-0.4} max={0.4} step={0.02}
                onChange={(v) => setHairFlow((h) => ({ ...h, cowlickPhi: v }))} />
              <div>
                <p className="text-[11px] text-slate-400 mb-1">渦の回転方向</p>
                <div className="flex gap-1.5">
                  <button
                    onClick={() => setHairFlow((h) => ({ ...h, cowlickDir: "cw" }))}
                    className={`flex-1 flex items-center justify-center gap-1 text-[11px] px-2 py-1.5 rounded-md border ${hairFlow.cowlickDir === "cw" ? "border-emerald-500 bg-emerald-500/15 text-emerald-300" : "border-slate-700 text-slate-400"}`}
                  ><RotateCw className="w-3.5 h-3.5" />時計回り</button>
                  <button
                    onClick={() => setHairFlow((h) => ({ ...h, cowlickDir: "ccw" }))}
                    className={`flex-1 flex items-center justify-center gap-1 text-[11px] px-2 py-1.5 rounded-md border ${hairFlow.cowlickDir === "ccw" ? "border-indigo-500 bg-indigo-500/15 text-indigo-300" : "border-slate-700 text-slate-400"}`}
                  ><RotateCcw className="w-3.5 h-3.5" />反時計回り</button>
                </div>
              </div>

              <div className="h-px bg-slate-800 my-2" />
              <div>
                <p className="text-[11px] text-slate-400 mb-1">ネープの生え癖</p>
                <SelectRow value={hairFlow.napeGrowth} onChange={(v) => setHairFlow((h) => ({ ...h, napeGrowth: v }))}
                  options={[["up", "浮き癖"], ["center", "真ん中に寄る"], ["down", "下向き"], ["outward", "上向き"]]} />
              </div>
              <div>
                <p className="text-[11px] text-slate-400 mb-1">前髪・顔周りの生え癖</p>
                <SelectRow value={hairFlow.frontGrowth} onChange={(v) => setHairFlow((h) => ({ ...h, frontGrowth: v }))}
                  options={[["left", "左割れ"], ["right", "右割れ"], ["up", "浮き上がり"], ["center", "センター分け"]]} />
              </div>
            </div>
          )}

          <div className="h-px bg-slate-800" />
          <button
            onClick={resetAll}
            className="w-full flex items-center justify-center gap-1.5 text-xs px-3 py-2 rounded-md border border-slate-700 text-slate-400 hover:text-slate-200 hover:border-slate-600"
          >
            <RefreshCw className="w-3.5 h-3.5" />すべて初期値に戻す
          </button>
        </aside>
      </div>

      {printOpen && (
        <PrintReport
          captures={captures}
          sections={sections}
          skull={skull}
          hairline={hairline}
          hairFlow={hairFlow}
          curl={curl}
          ironSize={ironSize}
          frontPhoto={frontPhoto}
          notes={notes}
          setNotes={setNotes}
          onClose={() => setPrintOpen(false)}
        />
      )}
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* Small UI helpers                                                    */
/* ------------------------------------------------------------------ */

function SliderRow({ label, value, min, max, step = 1, unit = "", leftLabel, rightLabel, onChange, help }) {
  return (
    <div>
      <div className="flex items-center justify-between mb-1">
        <span className="text-[11px] text-slate-400">{label}</span>
        <span className="text-[11px] text-emerald-400 font-medium tabular-nums">
          {Math.round(value * 100) / 100}{unit}
        </span>
      </div>
      <input
        type="range" min={min} max={max} step={step} value={value}
        onChange={(e) => onChange(Number(e.target.value))}
        className="w-full accent-indigo-500"
      />
      {(leftLabel || rightLabel) && (
        <div className="flex justify-between text-[10px] text-slate-600 mt-0.5">
          <span>{leftLabel}</span><span>{rightLabel}</span>
        </div>
      )}
      {help && <p className="text-[10px] text-slate-600 mt-0.5">{help}</p>}
    </div>
  );
}

function SelectRow({ value, onChange, options }) {
  return (
    <div className="flex flex-wrap gap-1.5">
      {options.map(([id, label]) => (
        <button
          key={id}
          onClick={() => onChange(id)}
          className={`text-[11px] px-2.5 py-1 rounded-md border ${
            value === id ? "border-emerald-500 bg-emerald-500/15 text-emerald-300" : "border-slate-700 text-slate-400 hover:border-slate-600"
          }`}
        >
          {label}
        </button>
      ))}
    </div>
  );
}

function PrintReport({ captures, sections, skull, hairline, hairFlow, curl, ironSize, frontPhoto, notes, setNotes, onClose }) {
  const sectionRows = Object.entries(sections);
  return (
    <div className="fixed inset-0 z-50 bg-slate-950/95 overflow-y-auto print:bg-white print:static">
      <div className="max-w-[900px] mx-auto p-4 print:p-0">
        <div className="flex items-center justify-between mb-3 print:hidden">
          <p className="text-sm text-slate-300">印刷プレビュー（A4）</p>
          <div className="flex gap-2">
            <button onClick={() => window.print()} className="text-xs px-3 py-1.5 rounded-md bg-indigo-600 hover:bg-indigo-500 text-white flex items-center gap-1.5">
              <Printer className="w-3.5 h-3.5" />印刷 / PDF保存
            </button>
            <button onClick={onClose} className="text-xs px-3 py-1.5 rounded-md border border-slate-700 text-slate-300 flex items-center gap-1.5">
              <X className="w-3.5 h-3.5" />閉じる
            </button>
          </div>
        </div>

        <div className="bg-white text-slate-900 rounded-lg print:rounded-none p-6 print:p-8 a4-sheet">
          <div className="flex items-center justify-between border-b-2 border-slate-800 pb-2 mb-3">
            <h2 className="text-lg font-bold">カット展開図 カルテ</h2>
            <span className="text-xs text-slate-500">{new Date().toLocaleDateString("ja-JP")} ・ アイロン: {ironSize === "straight" ? "ストレート" : ironSize + "mm"} ・ カール補正 {curl}%</span>
          </div>

          <div className="grid grid-cols-3 gap-2 mb-3">
            {frontPhoto && <img src={frontPhoto} className="w-full h-24 object-cover rounded border" alt="正面" />}
            {captures.slice(0, frontPhoto ? 2 : 3).map((c, i) => (
              <img key={i} src={c.src} className="w-full h-24 object-cover rounded border bg-slate-100" alt={c.label} />
            ))}
          </div>

          <div className="grid grid-cols-3 gap-2 mb-4">
            {captures.map((c, i) => (
              <div key={i} className="text-center">
                <img src={c.src} className="w-full h-28 object-cover rounded border bg-slate-100" alt={c.label} />
                <p className="text-[10px] text-slate-500 mt-0.5">{c.label}</p>
              </div>
            ))}
          </div>

          <table className="w-full text-[11px] border-collapse mb-3">
            <thead>
              <tr className="bg-slate-100">
                <th className="border border-slate-300 px-2 py-1 text-left">セクション</th>
                <th className="border border-slate-300 px-2 py-1">引き出し角度</th>
                <th className="border border-slate-300 px-2 py-1">オーバーダイレクション</th>
                <th className="border border-slate-300 px-2 py-1">スライス傾斜</th>
                <th className="border border-slate-300 px-2 py-1">長さ</th>
              </tr>
            </thead>
            <tbody>
              {sectionRows.map(([id, v]) => (
                <tr key={id}>
                  <td className="border border-slate-300 px-2 py-1 font-medium">{SECTIONS.find((s) => s.id === id)?.label}</td>
                  <td className="border border-slate-300 px-2 py-1 text-center">{v.elevation}°</td>
                  <td className="border border-slate-300 px-2 py-1 text-center">{v.overDirection}°</td>
                  <td className="border border-slate-300 px-2 py-1 text-center">{v.sliceTilt}°</td>
                  <td className="border border-slate-300 px-2 py-1 text-center">{v.length}cm</td>
                </tr>
              ))}
            </tbody>
          </table>

          <div className="grid grid-cols-2 gap-3 text-[11px] mb-3">
            <div className="border border-slate-300 rounded p-2">
              <p className="font-semibold mb-1">骨格補正</p>
              <p>ハチ張り: {skull.hachi.toFixed(2)} ／ 絶壁・丸み: {skull.roundness.toFixed(2)} ／ トップ高さ: {skull.topHeight.toFixed(2)}</p>
              <p>ネープ高さ: {hairline.napeHeight.toFixed(2)} ／ おでこの広さ: {hairline.foreheadWidth.toFixed(2)}</p>
            </div>
            <div className="border border-slate-300 rounded p-2">
              <p className="font-semibold mb-1">毛流・生え癖</p>
              <p>つむじ回転: {hairFlow.cowlickDir === "cw" ? "時計回り" : "反時計回り"}</p>
              <p>ネープ生え癖: {hairFlow.napeGrowth} ／ 前髪生え癖: {hairFlow.frontGrowth}</p>
            </div>
          </div>

          <div className="border border-slate-300 rounded p-2">
            <p className="font-semibold text-[11px] mb-1">カルテ・教育マニュアル用メモ</p>
            <textarea
              value={notes}
              onChange={(e) => setNotes(e.target.value)}
              rows={4}
              placeholder="施術メモ、注意点、次回来店時の指示など…"
              className="w-full text-[11px] border-none focus:outline-none resize-none print:border-t print:border-slate-200 print:pt-1"
            />
          </div>
        </div>
      </div>

      <style>{`
        @media print {
          @page { size: A4; margin: 10mm; }
          body { -webkit-print-color-adjust: exact; print-color-adjust: exact; }
        }
      `}</style>
    </div>
  );
}
