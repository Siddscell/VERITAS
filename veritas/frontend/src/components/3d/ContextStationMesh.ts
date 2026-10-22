import * as THREE from 'three';
import { STATIONS } from './types';

export class ContextStation {
  public group: THREE.Group;
  private pedestal: THREE.Mesh;
  private visualRing: THREE.Mesh;
  private audioRing: THREE.Mesh;
  private centralCore: THREE.Mesh;
  private bridgeRungs: THREE.LineSegments;
  private beacon: THREE.Mesh;
  private isHovered = false;
  private isSelected = false;

  constructor() {
    this.group = new THREE.Group();
    this.group.name = 'station_context';
    const cfg = STATIONS.context;
    this.group.position.set(...cfg.position);

    // 1. Hexagonal Cyber Pedestal
    const pedGeo = new THREE.CylinderGeometry(1.4, 1.6, 0.25, 6);
    const pedMat = new THREE.MeshStandardMaterial({
      color: 0x180928,
      emissive: 0xa855f7,
      emissiveIntensity: 0.22,
      metalness: 0.85,
      roughness: 0.25,
      flatShading: true,
    });
    this.pedestal = new THREE.Mesh(pedGeo, pedMat);
    this.pedestal.position.y = 0.125;
    this.group.add(this.pedestal);

    const pedWireGeo = new THREE.EdgesGeometry(pedGeo);
    const pedWireMat = new THREE.LineBasicMaterial({
      color: 0xa855f7,
      transparent: true,
      opacity: 0.6,
    });
    const pedWire = new THREE.LineSegments(pedWireGeo, pedWireMat);
    pedWire.position.copy(this.pedestal.position);
    this.group.add(pedWire);

    // 2. Interlocking Dual Stream Rings (Visual Stream & Audio Stream)
    const ringGeo = new THREE.TorusGeometry(0.75, 0.045, 8, 48);

    // Visual Stream Ring (Cyan-Blue tinted)
    const visualMat = new THREE.MeshStandardMaterial({
      color: 0x092b3a,
      emissive: 0x00f0ff,
      emissiveIntensity: 0.7,
      metalness: 0.9,
      roughness: 0.1,
    });
    this.visualRing = new THREE.Mesh(ringGeo, visualMat);
    this.visualRing.position.set(0, 1.35, 0);
    this.visualRing.rotation.set(0.6, 0.4, 0);
    this.group.add(this.visualRing);

    // Audio Stream Ring (Violet tinted)
    const audioMat = new THREE.MeshStandardMaterial({
      color: 0x220c3a,
      emissive: 0xa855f7,
      emissiveIntensity: 0.75,
      metalness: 0.9,
      roughness: 0.1,
    });
    this.audioRing = new THREE.Mesh(ringGeo, audioMat);
    this.audioRing.position.set(0, 1.35, 0);
    this.audioRing.rotation.set(-0.6, -0.4, 0);
    this.group.add(this.audioRing);

    // 3. Central Correlation Fusion Core Sphere
    const coreGeo = new THREE.SphereGeometry(0.28, 16, 16);
    const coreMat = new THREE.MeshStandardMaterial({
      color: 0xffffff,
      emissive: 0xd8b4fe,
      emissiveIntensity: 0.9,
      roughness: 0.1,
      metalness: 0.5,
    });
    this.centralCore = new THREE.Mesh(coreGeo, coreMat);
    this.centralCore.position.set(0, 1.35, 0);
    this.group.add(this.centralCore);

    // 4. Harmonic Bridge Conduits between the two rings
    const rungsGeo = new THREE.BufferGeometry();
    const rungsPoints: number[] = [];
    const count = 12;
    for (let i = 0; i < count; i++) {
      const angle = (i / count) * Math.PI * 2;
      const x1 = Math.cos(angle) * 0.7;
      const y1 = Math.sin(angle) * 0.5 + 1.35;
      const z1 = -0.3;

      const x2 = Math.cos(angle + 0.4) * 0.7;
      const y2 = Math.sin(angle + 0.4) * 0.5 + 1.35;
      const z2 = 0.3;

      rungsPoints.push(x1, y1, z1, x2, y2, z2);
    }
    rungsGeo.setAttribute('position', new THREE.Float32BufferAttribute(rungsPoints, 3));
    const rungsMat = new THREE.LineBasicMaterial({
      color: 0xc084fc,
      transparent: true,
      opacity: 0.5,
    });
    this.bridgeRungs = new THREE.LineSegments(rungsGeo, rungsMat);
    this.group.add(this.bridgeRungs);

    // 5. Floating Interactive Beacon
    const beaconGeo = new THREE.TetrahedronGeometry(0.2, 0);
    const beaconMat = new THREE.MeshStandardMaterial({
      color: 0xa855f7,
      emissive: 0xa855f7,
      emissiveIntensity: 0.9,
      roughness: 0.1,
    });
    this.beacon = new THREE.Mesh(beaconGeo, beaconMat);
    this.beacon.position.set(0, 2.45, 0);
    this.group.add(this.beacon);

    // Interactive tagging
    this.pedestal.userData = { isStation: true, stationId: 'context' };
    this.visualRing.userData = { isStation: true, stationId: 'context' };
    this.audioRing.userData = { isStation: true, stationId: 'context' };
    this.beacon.userData = { isStation: true, stationId: 'context' };
    this.centralCore.userData = { isStation: true, stationId: 'context' };
  }

  public setHovered(hovered: boolean) {
    this.isHovered = hovered;
  }

  public setSelected(selected: boolean) {
    this.isSelected = selected;
  }

  public update(delta: number, elapsed: number) {
    // Interlocking rings rotate in opposition
    this.visualRing.rotation.z += delta * 0.45;
    this.visualRing.rotation.y += delta * 0.25;

    this.audioRing.rotation.z -= delta * 0.45;
    this.audioRing.rotation.x += delta * 0.25;

    this.bridgeRungs.rotation.y += delta * 0.35;

    // Central correlation core pulse
    const pulse = 1.0 + Math.sin(elapsed * 3.2) * 0.1;
    this.centralCore.scale.set(pulse, pulse, pulse);

    // Beacon animation
    this.beacon.rotation.y += delta * 1.5;
    this.beacon.rotation.x += delta * 1.1;
    this.beacon.position.y = 2.45 + Math.sin(elapsed * 2.4) * 0.05;

    // Pedestal glow
    const pedMat = this.pedestal.material as THREE.MeshStandardMaterial;
    pedMat.emissiveIntensity = this.isSelected
      ? 0.8 + Math.sin(elapsed * 4.0) * 0.2
      : this.isHovered
      ? 0.6
      : 0.22;
  }

  public dispose() {
    this.pedestal.geometry.dispose();
    (this.pedestal.material as THREE.Material).dispose();
    this.visualRing.geometry.dispose();
    (this.visualRing.material as THREE.Material).dispose();
    this.audioRing.geometry.dispose();
    (this.audioRing.material as THREE.Material).dispose();
    this.centralCore.geometry.dispose();
    (this.centralCore.material as THREE.Material).dispose();
    this.bridgeRungs.geometry.dispose();
    (this.bridgeRungs.material as THREE.Material).dispose();
    this.beacon.geometry.dispose();
    (this.beacon.material as THREE.Material).dispose();
  }
}
