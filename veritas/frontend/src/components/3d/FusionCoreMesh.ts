import * as THREE from 'three';
import { STATIONS } from './types';

export class FusionCore {
  public group: THREE.Group;
  private coreMesh: THREE.Mesh;
  private wireCoreMesh: THREE.Mesh;
  private innerGlowMesh: THREE.Mesh;
  private ringX: THREE.Mesh;
  private ringY: THREE.Mesh;
  private ringZ: THREE.Mesh;
  private haloPoints: THREE.Points;
  private conduits: { line: THREE.Line; curve: THREE.CatmullRomCurve3 }[] = [];
  private conduitParticles: THREE.Points;
  private particleProgress: Float32Array;
  private isHovered = false;
  private isSelected = false;

  constructor() {
    this.group = new THREE.Group();
    this.group.name = 'fusion_core_group';
    this.group.position.set(0, 1.2, 0);

    // 1. Inner glowing faceted core
    const coreGeo = new THREE.IcosahedronGeometry(0.85, 2);
    const coreMat = new THREE.MeshStandardMaterial({
      color: 0x0a1420,
      emissive: 0xcc785c,
      emissiveIntensity: 0.35,
      metalness: 0.9,
      roughness: 0.2,
      flatShading: true,
    });
    this.coreMesh = new THREE.Mesh(coreGeo, coreMat);
    this.coreMesh.castShadow = true;
    this.group.add(this.coreMesh);

    // Wireframe cage over the core
    const wireGeo = new THREE.IcosahedronGeometry(0.92, 1);
    const wireMat = new THREE.MeshBasicMaterial({
      color: 0xffa07a,
      wireframe: true,
      transparent: true,
      opacity: 0.65,
    });
    this.wireCoreMesh = new THREE.Mesh(wireGeo, wireMat);
    this.group.add(this.wireCoreMesh);

    // Soft luminous inner core
    const innerGeo = new THREE.SphereGeometry(0.45, 16, 16);
    const innerMat = new THREE.MeshBasicMaterial({
      color: 0xffddcc,
      transparent: true,
      opacity: 0.8,
    });
    this.innerGlowMesh = new THREE.Mesh(innerGeo, innerMat);
    this.group.add(this.innerGlowMesh);

    // 2. Concentric Gyro Rings
    const createRing = (radius: number, tubeRadius: number, color: number) => {
      const ringGeo = new THREE.TorusGeometry(radius, tubeRadius, 8, 48);
      const ringMat = new THREE.MeshStandardMaterial({
        color: 0x111e2e,
        emissive: color,
        emissiveIntensity: 0.5,
        metalness: 0.95,
        roughness: 0.1,
      });
      return new THREE.Mesh(ringGeo, ringMat);
    };

    this.ringX = createRing(1.35, 0.025, 0x00f0ff);
    this.ringY = createRing(1.65, 0.025, 0x3b82f6);
    this.ringZ = createRing(1.95, 0.025, 0xcc785c);

    this.group.add(this.ringX);
    this.group.add(this.ringY);
    this.group.add(this.ringZ);

    // 3. Orbital Particle Cloud
    const haloCount = 180;
    const haloPos = new Float32Array(haloCount * 3);
    for (let i = 0; i < haloCount; i++) {
      const theta = Math.random() * Math.PI * 2;
      const phi = Math.acos(Math.random() * 2 - 1);
      const r = 1.8 + Math.random() * 0.8;
      haloPos[i * 3] = r * Math.sin(phi) * Math.cos(theta);
      haloPos[i * 3 + 1] = r * Math.sin(phi) * Math.sin(theta);
      haloPos[i * 3 + 2] = r * Math.cos(phi);
    }
    const haloGeo = new THREE.BufferGeometry();
    haloGeo.setAttribute('position', new THREE.BufferAttribute(haloPos, 3));
    const haloMat = new THREE.PointsMaterial({
      color: 0x00f0ff,
      size: 0.06,
      transparent: true,
      opacity: 0.75,
      blending: THREE.AdditiveBlending,
    });
    this.haloPoints = new THREE.Points(haloGeo, haloMat);
    this.group.add(this.haloPoints);

    // 4. Conduits from the 4 stations to the fusion core
    const conduitColorMap: Record<string, number> = {
      spatial: 0x00f0ff,
      temporal: 0x3b82f6,
      audio: 0x10b981,
      context: 0xa855f7,
    };

    const particlePositions: number[] = [];
    const particleColors: number[] = [];
    const stationKeys = ['spatial', 'temporal', 'audio', 'context'] as const;

    stationKeys.forEach((key) => {
      const station = STATIONS[key];
      const start = new THREE.Vector3(...station.position);
      start.y += 1.0;
      const end = new THREE.Vector3(0, 1.2, 0);

      // Curved spline arching slightly upward
      const mid = new THREE.Vector3(
        (start.x + end.x) * 0.5,
        Math.max(start.y, end.y) + 0.6,
        (start.z + end.z) * 0.5
      );

      const curve = new THREE.CatmullRomCurve3([start, mid, end]);
      const points = curve.getPoints(50);
      const lineGeo = new THREE.BufferGeometry().setFromPoints(points);

      const colorHex = conduitColorMap[key];
      const lineMat = new THREE.LineBasicMaterial({
        color: colorHex,
        transparent: true,
        opacity: 0.28,
        blending: THREE.AdditiveBlending,
      });

      const line = new THREE.Line(lineGeo, lineMat);
      this.conduits.push({ line, curve });
      // Add line to scene via parent later or directly in group relative coords
      // Since line coordinates are world-space, we offset them to group space
      line.position.set(0, -1.2, 0);
      this.group.add(line);

      // Create stream particles for this conduit
      const particlesPerConduit = 16;
      for (let p = 0; p < particlesPerConduit; p++) {
        particlePositions.push(0, 0, 0);
        const c = new THREE.Color(colorHex);
        particleColors.push(c.r, c.g, c.b);
      }
    });

    const totalConduitParticles = stationKeys.length * 16;
    this.particleProgress = new Float32Array(totalConduitParticles);
    for (let i = 0; i < totalConduitParticles; i++) {
      this.particleProgress[i] = (i % 16) / 16;
    }

    const streamGeo = new THREE.BufferGeometry();
    streamGeo.setAttribute(
      'position',
      new THREE.Float32BufferAttribute(particlePositions, 3)
    );
    streamGeo.setAttribute(
      'color',
      new THREE.Float32BufferAttribute(particleColors, 3)
    );

    const streamMat = new THREE.PointsMaterial({
      size: 0.12,
      vertexColors: true,
      transparent: true,
      opacity: 0.9,
      blending: THREE.AdditiveBlending,
    });

    this.conduitParticles = new THREE.Points(streamGeo, streamMat);
    this.conduitParticles.position.set(0, -1.2, 0);
    this.group.add(this.conduitParticles);

    // Tag the core for raycasting detection
    this.coreMesh.userData = { isStation: true, stationId: 'fusion' };
    this.wireCoreMesh.userData = { isStation: true, stationId: 'fusion' };
  }

  public setHovered(hovered: boolean) {
    this.isHovered = hovered;
  }

  public setSelected(selected: boolean) {
    this.isSelected = selected;
  }

  public update(delta: number, elapsed: number) {
    // Gyro ring rotations
    this.ringX.rotation.x += delta * 0.45;
    this.ringX.rotation.y += delta * 0.25;

    this.ringY.rotation.y += delta * 0.35;
    this.ringY.rotation.z += delta * 0.55;

    this.ringZ.rotation.z += delta * 0.28;
    this.ringZ.rotation.x += delta * 0.4;

    // Core breathing and rotation
    const baseScale = this.isSelected ? 1.25 : this.isHovered ? 1.15 : 1.0;
    const pulse = 1.0 + Math.sin(elapsed * 2.2) * 0.035;
    const currentScale = baseScale * pulse;
    this.coreMesh.scale.setScalar(currentScale);
    this.wireCoreMesh.scale.setScalar(currentScale * 1.05);

    this.coreMesh.rotation.y += delta * 0.2;
    this.wireCoreMesh.rotation.y -= delta * 0.3;
    this.wireCoreMesh.rotation.x += delta * 0.15;

    this.haloPoints.rotation.y += delta * 0.15;
    this.haloPoints.rotation.x -= delta * 0.08;

    // Dynamic emissive intensity
    const mat = this.coreMesh.material as THREE.MeshStandardMaterial;
    if (this.isHovered || this.isSelected) {
      mat.emissiveIntensity = 0.8 + Math.sin(elapsed * 4.0) * 0.2;
    } else {
      mat.emissiveIntensity = 0.35 + Math.sin(elapsed * 1.5) * 0.08;
    }

    // Animate evidence particles flowing from stations to fusion core
    const posAttr = this.conduitParticles.geometry.attributes.position;
    const positions = posAttr.array as Float32Array;

    let pIndex = 0;
    const stationKeysCount = 4;
    const perConduit = 16;

    for (let c = 0; c < stationKeysCount; c++) {
      const curve = this.conduits[c].curve;
      for (let i = 0; i < perConduit; i++) {
        // Increment progress towards core (0 -> 1)
        this.particleProgress[pIndex] += delta * 0.28;
        if (this.particleProgress[pIndex] > 1.0) {
          this.particleProgress[pIndex] -= 1.0;
        }

        const point = curve.getPoint(this.particleProgress[pIndex]);
        positions[pIndex * 3] = point.x;
        positions[pIndex * 3 + 1] = point.y;
        positions[pIndex * 3 + 2] = point.z;

        pIndex++;
      }
    }
    posAttr.needsUpdate = true;
  }

  public dispose() {
    this.coreMesh.geometry.dispose();
    (this.coreMesh.material as THREE.Material).dispose();
    this.wireCoreMesh.geometry.dispose();
    (this.wireCoreMesh.material as THREE.Material).dispose();
    this.innerGlowMesh.geometry.dispose();
    (this.innerGlowMesh.material as THREE.Material).dispose();
    this.ringX.geometry.dispose();
    (this.ringX.material as THREE.Material).dispose();
    this.ringY.geometry.dispose();
    (this.ringY.material as THREE.Material).dispose();
    this.ringZ.geometry.dispose();
    (this.ringZ.material as THREE.Material).dispose();
    this.haloPoints.geometry.dispose();
    (this.haloPoints.material as THREE.Material).dispose();
    this.conduitParticles.geometry.dispose();
    (this.conduitParticles.material as THREE.Material).dispose();
    this.conduits.forEach((c) => {
      c.line.geometry.dispose();
      (c.line.material as THREE.Material).dispose();
    });
  }
}
