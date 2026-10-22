import * as THREE from 'three';
import { STATIONS } from './types';

export class TemporalStation {
  public group: THREE.Group;
  private pedestal: THREE.Mesh;
  private framePlanes: THREE.Group;
  private vectorLines: THREE.LineSegments;
  private beacon: THREE.Mesh;
  private isHovered = false;
  private isSelected = false;

  constructor() {
    this.group = new THREE.Group();
    this.group.name = 'station_temporal';
    const cfg = STATIONS.temporal;
    this.group.position.set(...cfg.position);

    // 1. Hexagonal Cyber Pedestal
    const pedGeo = new THREE.CylinderGeometry(1.4, 1.6, 0.25, 6);
    const pedMat = new THREE.MeshStandardMaterial({
      color: 0x0a1428,
      emissive: 0x3b82f6,
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
      color: 0x3b82f6,
      transparent: true,
      opacity: 0.6,
    });
    const pedWire = new THREE.LineSegments(pedWireGeo, pedWireMat);
    pedWire.position.copy(this.pedestal.position);
    this.group.add(pedWire);

    // 2. Sequential Frame Planes (R3D-18 Temporal Slices)
    this.framePlanes = new THREE.Group();
    this.framePlanes.position.set(0, 1.35, 0);

    const numPlanes = 5;
    const planeGeo = new THREE.PlaneGeometry(1.2, 0.8);
    const planeEdgesGeo = new THREE.EdgesGeometry(planeGeo);
    const vectorPoints: number[] = [];

    for (let i = 0; i < numPlanes; i++) {
      const zOffset = (i - (numPlanes - 1) / 2) * 0.42;
      const xOffset = (i - (numPlanes - 1) / 2) * -0.15;
      const yOffset = (i - (numPlanes - 1) / 2) * 0.08;

      const planeMat = new THREE.MeshStandardMaterial({
        color: 0x091b38,
        emissive: 0x1d4ed8,
        emissiveIntensity: 0.2 + i * 0.08,
        transparent: true,
        opacity: 0.55 + i * 0.08,
        side: THREE.DoubleSide,
        roughness: 0.2,
        metalness: 0.8,
      });

      const plane = new THREE.Mesh(planeGeo, planeMat);
      plane.position.set(xOffset, yOffset, zOffset);
      plane.rotation.y = -0.35;
      this.framePlanes.add(plane);

      const edgeMat = new THREE.LineBasicMaterial({
        color: 0x60a5fa,
        transparent: true,
        opacity: 0.8,
      });
      const edge = new THREE.LineSegments(planeEdgesGeo, edgeMat);
      edge.position.copy(plane.position);
      edge.rotation.copy(plane.rotation);
      this.framePlanes.add(edge);

      // Connect key points between consecutive planes to represent temporal motion vectors
      if (i > 0) {
        const prevZ = (i - 1 - (numPlanes - 1) / 2) * 0.42;
        const prevX = (i - 1 - (numPlanes - 1) / 2) * -0.15;
        const prevY = (i - 1 - (numPlanes - 1) / 2) * 0.08;

        // Vector 1 (center)
        vectorPoints.push(prevX, prevY, prevZ, xOffset, yOffset, zOffset);
        // Vector 2 (corner top-right)
        vectorPoints.push(
          prevX + 0.4, prevY + 0.25, prevZ,
          xOffset + 0.4, yOffset + 0.25, zOffset
        );
        // Vector 3 (corner bottom-left)
        vectorPoints.push(
          prevX - 0.4, prevY - 0.25, prevZ,
          xOffset - 0.4, yOffset - 0.25, zOffset
        );
      }
    }

    const vectorGeo = new THREE.BufferGeometry();
    vectorGeo.setAttribute('position', new THREE.Float32BufferAttribute(vectorPoints, 3));
    const vectorMat = new THREE.LineBasicMaterial({
      color: 0x93c5fd,
      transparent: true,
      opacity: 0.7,
      linewidth: 1.5,
    });
    this.vectorLines = new THREE.LineSegments(vectorGeo, vectorMat);
    this.framePlanes.add(this.vectorLines);
    this.group.add(this.framePlanes);

    // 3. Floating Beacon
    const beaconGeo = new THREE.DodecahedronGeometry(0.18, 0);
    const beaconMat = new THREE.MeshStandardMaterial({
      color: 0x3b82f6,
      emissive: 0x3b82f6,
      emissiveIntensity: 0.85,
      roughness: 0.1,
    });
    this.beacon = new THREE.Mesh(beaconGeo, beaconMat);
    this.beacon.position.set(0, 2.5, 0);
    this.group.add(this.beacon);

    // Interactive tagging
    this.pedestal.userData = { isStation: true, stationId: 'temporal' };
    this.beacon.userData = { isStation: true, stationId: 'temporal' };
    this.framePlanes.children.forEach((c) => {
      c.userData = { isStation: true, stationId: 'temporal' };
    });
  }

  public setHovered(hovered: boolean) {
    this.isHovered = hovered;
  }

  public setSelected(selected: boolean) {
    this.isSelected = selected;
  }

  public update(delta: number, elapsed: number) {
    // Gentle floating breathing animation of temporal planes
    this.framePlanes.position.y = 1.35 + Math.sin(elapsed * 1.8) * 0.04;

    // Hover / selection expansion
    const targetScale = this.isSelected ? 1.1 : this.isHovered ? 1.05 : 1.0;
    this.framePlanes.scale.lerp(new THREE.Vector3(targetScale, targetScale, targetScale), 0.1);

    // Beacon rotation
    this.beacon.rotation.y -= delta * 1.2;
    this.beacon.rotation.z += delta * 0.9;
    this.beacon.position.y = 2.5 + Math.sin(elapsed * 2.2 + 1.0) * 0.05;

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
    this.vectorLines.geometry.dispose();
    (this.vectorLines.material as THREE.Material).dispose();
    this.framePlanes.children.forEach((c) => {
      if (c instanceof THREE.Mesh) {
        c.geometry.dispose();
        (c.material as THREE.Material).dispose();
      }
    });
  }
}
