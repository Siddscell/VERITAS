import * as THREE from 'three';

export interface EnvironmentController {
  group: THREE.Group;
  stationLights: { [id: string]: THREE.PointLight };
  update: (dt: number, elapsed: number) => void;
  dispose: () => void;
}

export function createSceneEnvironment(scene: THREE.Scene): EnvironmentController {
  const group = new THREE.Group();
  group.name = 'EnvironmentGroup';

  // 1. Atmospheric Fog (deep midnight void)
  scene.background = new THREE.Color(0x040812);
  scene.fog = new THREE.FogExp2(0x040812, 0.024);

  // 2. Global Ambient Lighting (midnight cyan/navy)
  const ambientLight = new THREE.AmbientLight(0x0d1b2a, 0.85);
  group.add(ambientLight);

  // 3. Directional Key Light (cool cyber cyan from upper right)
  const keyLight = new THREE.DirectionalLight(0x38bdf8, 1.4);
  keyLight.position.set(12, 22, 14);
  group.add(keyLight);

  // 4. Directional Rim Light (deep ultraviolet from back left)
  const rimLight = new THREE.DirectionalLight(0x818cf8, 0.9);
  rimLight.position.set(-14, 16, -12);
  group.add(rimLight);

  // 5. Central Fusion Core Accent Light
  const coreLight = new THREE.PointLight(0x00f0ff, 2.2, 16, 1.8);
  coreLight.position.set(0, 1.2, 0);
  group.add(coreLight);

  // 6. Station Dedicated Spot/Point Lights
  const stationLights: { [id: string]: THREE.PointLight } = {
    spatial: new THREE.PointLight(0x00f0ff, 1.5, 10, 2),
    temporal: new THREE.PointLight(0x6366f1, 1.5, 10, 2),
    audio: new THREE.PointLight(0xa855f7, 1.5, 10, 2),
    context: new THREE.PointLight(0x10b981, 1.5, 10, 2),
  };

  stationLights.spatial.position.set(-6, 2.5, -4);
  stationLights.temporal.position.set(6, 2.5, -4);
  stationLights.audio.position.set(-6, 2.5, 4);
  stationLights.context.position.set(6, 2.5, 4);

  Object.values(stationLights).forEach(light => group.add(light));

  // 7. Cybernetic Laboratory Floor Grid
  const gridHelper = new THREE.GridHelper(36, 36, 0x1e3a5f, 0x0c1e33);
  gridHelper.position.y = -0.02;
  group.add(gridHelper);

  // Concentric Sector Marker Rings on the floor
  const ringGeometries: THREE.BufferGeometry[] = [];
  const ringMaterials: THREE.Material[] = [];
  const ringRadii = [3.2, 7.5, 12];

  ringRadii.forEach((radius, idx) => {
    const ringGeo = new THREE.RingGeometry(radius, radius + 0.04, 64);
    ringGeometries.push(ringGeo);
    const ringMat = new THREE.MeshBasicMaterial({
      color: idx === 0 ? 0x00f0ff : 0x1e40af,
      transparent: true,
      opacity: idx === 0 ? 0.35 : 0.18,
      side: THREE.DoubleSide,
    });
    ringMaterials.push(ringMat);
    const ringMesh = new THREE.Mesh(ringGeo, ringMat);
    ringMesh.rotation.x = -Math.PI / 2;
    ringMesh.position.y = -0.01;
    group.add(ringMesh);
  });

  // 8. Atmospheric Cybernetic Floating Particles
  const particleCount = 1000;
  const particleGeo = new THREE.BufferGeometry();
  const positions = new Float32Array(particleCount * 3);
  const baseSpeeds = new Float32Array(particleCount);

  for (let i = 0; i < particleCount; i++) {
    positions[i * 3 + 0] = (Math.random() - 0.5) * 36;
    positions[i * 3 + 1] = Math.random() * 14;
    positions[i * 3 + 2] = (Math.random() - 0.5) * 36;
    baseSpeeds[i] = 0.2 + Math.random() * 0.4;
  }

  particleGeo.setAttribute('position', new THREE.BufferAttribute(positions, 3));

  const particleMat = new THREE.PointsMaterial({
    color: 0x38bdf8,
    size: 0.045,
    transparent: true,
    opacity: 0.55,
    blending: THREE.AdditiveBlending,
    depthWrite: false,
  });

  const particleSystem = new THREE.Points(particleGeo, particleMat);
  group.add(particleSystem);

  scene.add(group);

  // Animate particles & dynamic lighting pulse
  const update = (dt: number, elapsed: number) => {
    // Pulse central core light gently
    coreLight.intensity = 2.0 + Math.sin(elapsed * 2.5) * 0.5;

    // Pulse station lights with subtle phase offset
    stationLights.spatial.intensity = 1.3 + Math.sin(elapsed * 1.8 + 0.0) * 0.3;
    stationLights.temporal.intensity = 1.3 + Math.sin(elapsed * 1.8 + 1.2) * 0.3;
    stationLights.audio.intensity = 1.3 + Math.sin(elapsed * 1.8 + 2.4) * 0.3;
    stationLights.context.intensity = 1.3 + Math.sin(elapsed * 1.8 + 3.6) * 0.3;

    // Drift particles gently upward
    const posAttr = particleGeo.getAttribute('position') as THREE.BufferAttribute;
    const array = posAttr.array as Float32Array;

    for (let i = 0; i < particleCount; i++) {
      let y = array[i * 3 + 1];
      y += baseSpeeds[i] * dt * 0.6;
      if (y > 14) {
        y = 0.1;
      }
      array[i * 3 + 1] = y;

      // Slight drift along X/Z
      array[i * 3 + 0] += Math.sin(elapsed * 0.3 + i) * dt * 0.05;
    }
    posAttr.needsUpdate = true;
  };

  const dispose = () => {
    scene.remove(group);
    gridHelper.dispose();
    particleGeo.dispose();
    particleMat.dispose();
    ringGeometries.forEach(g => g.dispose());
    ringMaterials.forEach(m => m.dispose());
  };

  return {
    group,
    stationLights,
    update,
    dispose,
  };
}
