import * as THREE from 'three';
import { STATIONS } from './types';

export class AudioStation {
  public group: THREE.Group;
  private pedestal: THREE.Mesh;
  private spectrumBars: THREE.Mesh[] = [];
  private soundWaveRings: THREE.Mesh[] = [];
  private beacon: THREE.Mesh;
  private isHovered = false;
  private isSelected = false;

  constructor() {
    this.group = new THREE.Group();
    this.group.name = 'station_audio';
    const cfg = STATIONS.audio;
    this.group.position.set(...cfg.position);

    // 1. Cyber Pedestal
    const pedGeo = new THREE.CylinderGeometry(1.4, 1.6, 0.25, 6);
    const pedMat = new THREE.MeshStandardMaterial({
      color: 0x061e1a,
      emissive: 0x10b981,
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
      color: 0x10b981,
      transparent: true,
      opacity: 0.6,
    });
    const pedWire = new THREE.LineSegments(pedWireGeo, pedWireMat);
    pedWire.position.copy(this.pedestal.position);
    this.group.add(pedWire);

    // 2. 3D Radial Mel-Spectrogram Frequency Spectrum Array (24 bars in a circular arc)
    const numBars = 24;
    const radius = 0.95;
    const barGeo = new THREE.BoxGeometry(0.08, 1.0, 0.08);

    for (let i = 0; i < numBars; i++) {
      const angle = (i / numBars) * Math.PI * 2;
      const x = Math.cos(angle) * radius;
      const z = Math.sin(angle) * radius;

      const barMat = new THREE.MeshStandardMaterial({
        color: 0x064e3b,
        emissive: 0x34d399,
        emissiveIntensity: 0.6,
        roughness: 0.2,
        metalness: 0.8,
      });

      const bar = new THREE.Mesh(barGeo, barMat);
      bar.position.set(x, 0.75, z);
      bar.rotation.y = -angle;
      this.spectrumBars.push(bar);
      this.group.add(bar);
    }

    // 3. Floating Concentric Sound Wave Rings
    for (let r = 0; r < 2; r++) {
      const ringGeo = new THREE.TorusGeometry(0.6 + r * 0.35, 0.015, 6, 32);
      const ringMat = new THREE.MeshBasicMaterial({
        color: 0x10b981,
        transparent: true,
        opacity: 0.5 - r * 0.15,
      });
      const ring = new THREE.Mesh(ringGeo, ringMat);
      ring.position.y = 1.35;
      ring.rotation.x = Math.PI / 2;
      this.soundWaveRings.push(ring);
      this.group.add(ring);
    }

    // 4. Floating Beacon
    const beaconGeo = new THREE.ConeGeometry(0.18, 0.35, 6);
    const beaconMat = new THREE.MeshStandardMaterial({
      color: 0x10b981,
      emissive: 0x10b981,
      emissiveIntensity: 0.85,
      roughness: 0.1,
    });
    this.beacon = new THREE.Mesh(beaconGeo, beaconMat);
    this.beacon.position.set(0, 2.45, 0);
    this.group.add(this.beacon);

    // Interactive tagging
    this.pedestal.userData = { isStation: true, stationId: 'audio' };
    this.beacon.userData = { isStation: true, stationId: 'audio' };
    this.spectrumBars.forEach((b) => {
      b.userData = { isStation: true, stationId: 'audio' };
    });
  }

  public setHovered(hovered: boolean) {
    this.isHovered = hovered;
  }

  public setSelected(selected: boolean) {
    this.isSelected = selected;
  }

  public update(delta: number, elapsed: number) {
    // Dynamic acoustic spectrogram wave undulation
    const baseEnergy = this.isSelected ? 1.6 : this.isHovered ? 1.3 : 1.0;
    this.spectrumBars.forEach((bar, idx) => {
      // Simulate multi-band acoustic energy (bass, midrange, vocoder harmonic spikes)
      const freq1 = Math.sin(elapsed * 4.5 + idx * 0.45);
      const freq2 = Math.cos(elapsed * 2.8 + idx * 0.9);
      const rawHeight = 0.25 + Math.abs(freq1 * 0.5 + freq2 * 0.35) * baseEnergy;
      const height = Math.max(0.15, Math.min(rawHeight, 1.4));

      bar.scale.y = height;
      bar.position.y = 0.25 + height * 0.5;

      const mat = bar.material as THREE.MeshStandardMaterial;
      mat.emissiveIntensity = 0.4 + height * 0.5;
    });

    // Sound ring expansion
    this.soundWaveRings.forEach((ring, idx) => {
      ring.rotation.z += delta * (0.3 + idx * 0.2);
      const pulse = 1.0 + Math.sin(elapsed * 3.0 + idx) * 0.08;
      ring.scale.set(pulse, pulse, pulse);
    });

    // Beacon rotation
    this.beacon.rotation.y += delta * 1.8;
    this.beacon.position.y = 2.45 + Math.sin(elapsed * 2.0) * 0.05;

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
    this.beacon.geometry.dispose();
    (this.beacon.material as THREE.Material).dispose();
    this.spectrumBars.forEach((b) => {
      b.geometry.dispose();
      (b.material as THREE.Material).dispose();
    });
    this.soundWaveRings.forEach((r) => {
      r.geometry.dispose();
      (r.material as THREE.Material).dispose();
    });
  }
}
