import React, { useEffect, useRef, useState } from 'react';
import * as THREE from 'three';
import { StationId, STATIONS, OVERVIEW_CAMERA } from './types';
import { createSceneEnvironment, EnvironmentController } from './SceneEnvironment';
import { FusionCore } from './FusionCoreMesh';
import { SpatialStation } from './SpatialStationMesh';
import { TemporalStation } from './TemporalStationMesh';
import { AudioStation } from './AudioStationMesh';
import { ContextStation } from './ContextStationMesh';

interface ForensicsLabCanvasProps {
  activeStationId: StationId;
  onSelectStation: (id: StationId) => void;
  onHoverStation?: (id: StationId | null) => void;
  className?: string;
}

export const ForensicsLabCanvas: React.FC<ForensicsLabCanvasProps> = ({
  activeStationId,
  onSelectStation,
  onHoverStation,
  className = '',
}) => {
  const containerRef = useRef<HTMLDivElement>(null);
  const [webGlSupported, setWebGlSupported] = useState(true);
  const [isLoaded, setIsLoaded] = useState(false);

  // References for cross-frame coordination
  const activeStationRef = useRef<StationId>(activeStationId);
  activeStationRef.current = activeStationId;

  const onSelectStationRef = useRef(onSelectStation);
  onSelectStationRef.current = onSelectStation;

  const onHoverStationRef = useRef(onHoverStation);
  onHoverStationRef.current = onHoverStation;

  // Normalized mouse coords [-1, 1]
  const mouseRef = useRef<{ x: number; y: number; moved: boolean }>({ x: 0, y: 0, moved: false });

  // WebGL Feature test helper
  const checkWebGL = () => {
    try {
      const testCanvas = document.createElement('canvas');
      return !!(
        window.WebGLRenderingContext &&
        (testCanvas.getContext('webgl') || testCanvas.getContext('experimental-webgl'))
      );
    } catch {
      return false;
    }
  };

  useEffect(() => {
    if (!checkWebGL()) {
      setWebGlSupported(false);
      return;
    }

    const container = containerRef.current;
    if (!container) return;

    const width = container.clientWidth || window.innerWidth;
    const height = container.clientHeight || window.innerHeight;

    // 1. Scene & Renderer Setup
    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(46, width / height, 0.1, 90);

    const initialPos = OVERVIEW_CAMERA.position;
    const initialLookAt = OVERVIEW_CAMERA.lookAt;
    camera.position.set(initialPos[0], initialPos[1], initialPos[2]);
    camera.lookAt(new THREE.Vector3(...initialLookAt));

    const renderer = new THREE.WebGLRenderer({
      powerPreference: 'high-performance',
      antialias: true,
      alpha: false,
    });
    renderer.setSize(width, height);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 1.1;

    container.appendChild(renderer.domElement);
    renderer.domElement.style.outline = 'none';
    renderer.domElement.style.touchAction = 'none';

    // 2. Instantiate Stations and Controllers
    let envController: EnvironmentController | null = null;
    let fusionCore: FusionCore | null = null;
    let spatialStation: SpatialStation | null = null;
    let temporalStation: TemporalStation | null = null;
    let audioStation: AudioStation | null = null;
    let contextStation: ContextStation | null = null;

    try {
      envController = createSceneEnvironment(scene);

      fusionCore = new FusionCore();
      scene.add(fusionCore.group);

      spatialStation = new SpatialStation();
      scene.add(spatialStation.group);

      temporalStation = new TemporalStation();
      scene.add(temporalStation.group);

      audioStation = new AudioStation();
      scene.add(audioStation.group);

      contextStation = new ContextStation();
      scene.add(contextStation.group);
    } catch (err) {
      console.error('Failed to initialize 3D scene modules:', err);
      setWebGlSupported(false);
      return;
    }

    // 3. Raycasting Interactive Targets
    // Invisible hit spheres for reliable raycasting across devices
    const interactiveTargets: { id: StationId; sphere: THREE.Mesh }[] = [
      { id: 'spatial', sphere: new THREE.Mesh(new THREE.SphereGeometry(2.4, 8, 8), new THREE.MeshBasicMaterial({ visible: false })) },
      { id: 'temporal', sphere: new THREE.Mesh(new THREE.SphereGeometry(2.4, 8, 8), new THREE.MeshBasicMaterial({ visible: false })) },
      { id: 'audio', sphere: new THREE.Mesh(new THREE.SphereGeometry(2.4, 8, 8), new THREE.MeshBasicMaterial({ visible: false })) },
      { id: 'context', sphere: new THREE.Mesh(new THREE.SphereGeometry(2.4, 8, 8), new THREE.MeshBasicMaterial({ visible: false })) },
      { id: 'fusion', sphere: new THREE.Mesh(new THREE.SphereGeometry(2.6, 8, 8), new THREE.MeshBasicMaterial({ visible: false })) },
    ];

    interactiveTargets[0].sphere.position.set(-6, 1.2, -4);
    interactiveTargets[1].sphere.position.set(6, 1.2, -4);
    interactiveTargets[2].sphere.position.set(-6, 1.2, 4);
    interactiveTargets[3].sphere.position.set(6, 1.2, 4);
    interactiveTargets[4].sphere.position.set(0, 1.2, 0);

    interactiveTargets.forEach(t => scene.add(t.sphere));

    const raycaster = new THREE.Raycaster();
    const mouseVector = new THREE.Vector2();

    // Camera Motion interpolation state
    const currentCamPos = new THREE.Vector3(...initialPos);
    const currentCamLookAt = new THREE.Vector3(...initialLookAt);
    const targetCamPos = new THREE.Vector3(...initialPos);
    const targetCamLookAt = new THREE.Vector3(...initialLookAt);

    let hoveredStation: StationId | null = null;

    // Helper: update hover highlights
    const setHoverState = (stationId: StationId | null) => {
      if (stationId === hoveredStation) return;
      hoveredStation = stationId;

      spatialStation?.setHovered(stationId === 'spatial');
      temporalStation?.setHovered(stationId === 'temporal');
      audioStation?.setHovered(stationId === 'audio');
      contextStation?.setHovered(stationId === 'context');
      fusionCore?.setHovered(stationId === 'fusion');

      renderer.domElement.style.cursor = stationId ? 'pointer' : 'default';
      onHoverStationRef.current?.(stationId);
    };

    // 4. Pointer Events
    const onPointerMove = (e: MouseEvent) => {
      const rect = renderer.domElement.getBoundingClientRect();
      const clientX = e.clientX - rect.left;
      const clientY = e.clientY - rect.top;

      mouseVector.x = (clientX / rect.width) * 2 - 1;
      mouseVector.y = -(clientY / rect.height) * 2 + 1;

      mouseRef.current.x = mouseVector.x;
      mouseRef.current.y = mouseVector.y;
      mouseRef.current.moved = true;

      // Raycast against station hit spheres
      raycaster.setFromCamera(mouseVector, camera);
      const hitSpheres = interactiveTargets.map(t => t.sphere);
      const intersects = raycaster.intersectObjects(hitSpheres, false);

      if (intersects.length > 0) {
        const hitTarget = interactiveTargets.find(t => t.sphere === intersects[0].object);
        if (hitTarget) {
          setHoverState(hitTarget.id);
          return;
        }
      }
      setHoverState(null);
    };

    const onPointerClick = (e: MouseEvent) => {
      const rect = renderer.domElement.getBoundingClientRect();
      const clientX = e.clientX - rect.left;
      const clientY = e.clientY - rect.top;

      const clickVec = new THREE.Vector2(
        (clientX / rect.width) * 2 - 1,
        -(clientY / rect.height) * 2 + 1
      );

      raycaster.setFromCamera(clickVec, camera);
      const hitSpheres = interactiveTargets.map(t => t.sphere);
      const intersects = raycaster.intersectObjects(hitSpheres, false);

      if (intersects.length > 0) {
        const hitTarget = interactiveTargets.find(t => t.sphere === intersects[0].object);
        if (hitTarget) {
          onSelectStationRef.current(hitTarget.id);
        }
      }
    };

    const domElem = renderer.domElement;
    domElem.addEventListener('mousemove', onPointerMove, { passive: true });
    domElem.addEventListener('click', onPointerClick);

    // 5. Responsive Resizing
    const resizeObserver = new ResizeObserver((entries) => {
      for (const entry of entries) {
        const w = entry.contentRect.width;
        const h = entry.contentRect.height;
        if (w > 0 && h > 0) {
          camera.aspect = w / h;
          camera.updateProjectionMatrix();
          renderer.setSize(w, h);
        }
      }
    });
    resizeObserver.observe(container);

    // 6. Animation Loop with Parallax & Camera Rails
    let animationFrameId: number;
    let lastTime = performance.now();
    let isVisible = true;

    // Detect prefers-reduced-motion
    const reducedMotionQuery = window.matchMedia('(prefers-reduced-motion: reduce)');
    let prefersReducedMotion = reducedMotionQuery.matches;
    const onReducedMotionChange = (e: MediaQueryListEvent) => {
      prefersReducedMotion = e.matches;
    };
    reducedMotionQuery.addEventListener('change', onReducedMotionChange);

    // Visibility observer to pause RAF when off-screen
    const intersectionObserver = new IntersectionObserver(([entry]) => {
      isVisible = entry.isIntersecting;
    }, { threshold: 0.05 });
    intersectionObserver.observe(container);

    const animate = (currentTime: number) => {
      animationFrameId = requestAnimationFrame(animate);

      if (!isVisible) return;

      const dt = Math.min((currentTime - lastTime) / 1000, 0.1);
      lastTime = currentTime;
      const elapsed = currentTime * 0.001;

      // Update 3D Station Meshes
      envController?.update(dt, elapsed);
      fusionCore?.update(dt, elapsed);
      spatialStation?.update(dt, elapsed);
      temporalStation?.update(dt, elapsed);
      audioStation?.update(dt, elapsed);
      contextStation?.update(dt, elapsed);

      // Determine Target Camera Position and LookAt based on activeStationId
      const currentActive = activeStationRef.current;
      if (currentActive === 'overview') {
        targetCamPos.set(...OVERVIEW_CAMERA.position);
        targetCamLookAt.set(...OVERVIEW_CAMERA.lookAt);
      } else {
        const cfg = STATIONS[currentActive];
        if (cfg) {
          targetCamPos.set(...cfg.cameraPosition);
          targetCamLookAt.set(...cfg.targetLookAt);
        }
      }

      // Parallax effect: Subtle camera shift according to mouse position
      const parallaxFactor = prefersReducedMotion ? 0 : (currentActive === 'overview' ? 0.75 : 0.2);
      const targetWithParallaxX = targetCamPos.x + mouseRef.current.x * parallaxFactor;
      const targetWithParallaxY = targetCamPos.y + mouseRef.current.y * (parallaxFactor * 0.4);
      const targetWithParallaxZ = targetCamPos.z;

      // Smooth camera interpolation
      const lerpSpeed = prefersReducedMotion ? 8.0 : 3.0;
      currentCamPos.x += (targetWithParallaxX - currentCamPos.x) * Math.min(1, dt * lerpSpeed);
      currentCamPos.y += (targetWithParallaxY - currentCamPos.y) * Math.min(1, dt * lerpSpeed);
      currentCamPos.z += (targetWithParallaxZ - currentCamPos.z) * Math.min(1, dt * lerpSpeed);

      currentCamLookAt.x += (targetCamLookAt.x - currentCamLookAt.x) * Math.min(1, dt * lerpSpeed);
      currentCamLookAt.y += (targetCamLookAt.y - currentCamLookAt.y) * Math.min(1, dt * lerpSpeed);
      currentCamLookAt.z += (targetCamLookAt.z - currentCamLookAt.z) * Math.min(1, dt * lerpSpeed);

      camera.position.copy(currentCamPos);
      camera.lookAt(currentCamLookAt);

      renderer.render(scene, camera);
    };

    animationFrameId = requestAnimationFrame(animate);
    setIsLoaded(true);

    // 7. Cleanup Function
    return () => {
      cancelAnimationFrame(animationFrameId);
      resizeObserver.disconnect();
      intersectionObserver.disconnect();
      reducedMotionQuery.removeEventListener('change', onReducedMotionChange);

      domElem.removeEventListener('mousemove', onPointerMove);
      domElem.removeEventListener('click', onPointerClick);

      // Dispose interactive spheres
      interactiveTargets.forEach(t => {
        scene.remove(t.sphere);
        t.sphere.geometry.dispose();
        (t.sphere.material as THREE.Material).dispose();
      });

      // Dispose controllers
      envController?.dispose();
      if (fusionCore) {
        scene.remove(fusionCore.group);
        fusionCore.dispose();
      }
      if (spatialStation) {
        scene.remove(spatialStation.group);
        spatialStation.dispose();
      }
      if (temporalStation) {
        scene.remove(temporalStation.group);
        temporalStation.dispose();
      }
      if (audioStation) {
        scene.remove(audioStation.group);
        audioStation.dispose();
      }
      if (contextStation) {
        scene.remove(contextStation.group);
        contextStation.dispose();
      }

      renderer.dispose();
      if (domElem.parentElement === container) {
        container.removeChild(domElem);
      }
    };
  }, []);

  if (!webGlSupported) {
    return (
      <div className={`relative flex items-center justify-center p-8 bg-zinc-950/90 border border-cyan-500/30 rounded-xl overflow-hidden ${className}`}>
        <div className="absolute inset-0 bg-[radial-gradient(circle_at_center,rgba(6,182,212,0.1)_0%,transparent_70%)] pointer-events-none" />
        <div className="max-w-lg text-center z-10 space-y-4">
          <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full border border-cyan-500/40 bg-cyan-950/30 text-cyan-400 text-xs font-mono">
            <span>2D COMPATIBILITY FALLBACK ACTIVE</span>
          </div>
          <h3 className="text-xl font-bold text-white tracking-wide">
            VERITAS Forensic Multi-Agent Architecture
          </h3>
          <p className="text-sm text-zinc-400 leading-relaxed">
            WebGL is not hardware-accelerated or enabled on your display pipeline. You can still inspect all four forensics stations, review model architectures, and launch the live verification console.
          </p>
          <div className="grid grid-cols-2 gap-3 pt-2">
            {(Object.keys(STATIONS) as (keyof typeof STATIONS)[]).map((key) => {
              const s = STATIONS[key];
              return (
                <button
                  key={s.id}
                  onClick={() => onSelectStation(s.id)}
                  className="p-3 text-left rounded-lg border border-zinc-800 bg-zinc-900/60 hover:border-cyan-500/50 hover:bg-cyan-950/20 transition-all cursor-pointer"
                >
                  <div className="text-xs font-mono text-cyan-400">{s.badge}</div>
                  <div className="text-sm font-semibold text-zinc-200">{s.name}</div>
                  <div className="text-xs text-zinc-500 line-clamp-1">{s.modelName}</div>
                </button>
              );
            })}
          </div>
        </div>
      </div>
    );
  }

  return (
    <div
      ref={containerRef}
      className={`relative w-full h-full overflow-hidden select-none ${className}`}
      style={{ touchAction: 'none' }}
    >
      {!isLoaded && (
        <div className="absolute inset-0 flex items-center justify-center bg-zinc-950 z-20">
          <div className="flex flex-col items-center gap-3">
            <div className="w-10 h-10 border-2 border-cyan-500/30 border-t-cyan-400 rounded-full animate-spin" />
            <span className="text-xs font-mono text-cyan-400/80 tracking-widest uppercase">
              Initializing 3D Forensic Matrix...
            </span>
          </div>
        </div>
      )}
    </div>
  );
};
