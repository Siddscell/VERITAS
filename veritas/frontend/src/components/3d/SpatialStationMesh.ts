import * as THREE from 'three';
import { STATIONS } from './types';

export class SpatialStation {
  public group: THREE.Group;
  private pedestal: THREE.Mesh;
  private frameBorder: THREE.LineSegments;
  private faceWireframe: THREE.LineSegments;
  private scanLine: THREE.Line;
  private beacon: THREE.Mesh;
  private isHovered = false;
  private isSelected = false;
  private scanY = -0.6;
  private scanDirection = 1;

  constructor() {
    this.group = new THREE.Group();
    this.group.name = 'station_spatial';
    const cfg = STATIONS.spatial;
    this.group.position.set(...cfg.position);

    const cyanColor = new THREE.Color(cfg.accentColor);

    // 1. Cybernetic Hexagonal Pedestal
    const pedGeo = new THREE.CylinderGeometry(1.4, 1.6, 0.25, 6);
    const pedMat = new THREE.MeshStandardMaterial({
      color: 0x0c1420,
      emissive: 0x00f0ff,
      emissiveIntensity: 0.2,
      metalness: 0.85,
      roughness: 0.25,
      flatShading: true,
    });
    this.pedestal = new THREE.Mesh(pedGeo, pedMat);
    this.pedestal.position.y = 0.125;
    this.group.add(this.pedestal);

    // Pedestal edge wireframe
    const pedWireGeo = new THREE.EdgesGeometry(pedGeo);
    const pedWireMat = new THREE.LineBasicMaterial({
      color: 0x00f0ff,
      transparent: true,
      opacity: 0.6,
    });
    const pedWire = new THREE.LineSegments(pedWireGeo, pedWireMat);
    pedWire.position.copy(this.pedestal.position);
    this.group.add(pedWire);

    // 2. Holographic Video Frame (16:9 Aspect Ratio)
    const frameWidth = 2.0;
    const frameHeight = 1.35;
    const boxGeo = new THREE.BoxGeometry(frameWidth, frameHeight, 0.08);
    const boxEdges = new THREE.EdgesGeometry(boxGeo);
    const frameMat = new THREE.LineBasicMaterial({
      color: 0x00f0ff,
      linewidth: 2,
    });
    this.frameBorder = new THREE.LineSegments(boxEdges, frameMat);
    this.frameBorder.position.set(0, 1.45, 0);
    this.group.add(this.frameBorder);

    // Translucent glass backplane
    const backplaneGeo = new THREE.PlaneGeometry(frameWidth * 0.98, frameHeight * 0.98);
    const backplaneMat = new THREE.MeshStandardMaterial({
      color: 0x031826,
      transparent: true,
      opacity: 0.45,
      roughness: 0.1,
      metalness: 0.9,
      side: THREE.DoubleSide,
    });
    const backplane = new THREE.Mesh(backplaneGeo, backplaneMat);
    backplane.position.set(0, 1.45, 0);
    this.group.add(backplane);

    // 3. Low-Poly Facial Landmark Wireframe inside frame
    // Procedural stylized face mesh vertices (oval head, eyes, nose, mouth landmarks)
    const faceGeo = new THREE.BufferGeometry();
    const facePoints: number[] = [
      // Outer contour
      0, 0.45, 0.04, -0.22, 0.32, 0.03,
      -0.22, 0.32, 0.03, -0.28, 0.05, 0.02,
      -0.28, 0.05, 0.02, -0.24, -0.22, 0.02,
      -0.24, -0.22, 0.02, -0.12, -0.42, 0.03,
      -0.12, -0.42, 0.03, 0, -0.48, 0.04,
      0, -0.48, 0.04, 0.12, -0.42, 0.03,
      0.12, -0.42, 0.03, 0.24, -0.22, 0.02,
      0.24, -0.22, 0.02, 0.28, 0.05, 0.02,
      0.28, 0.05, 0.02, 0.22, 0.32, 0.03,
      0.22, 0.32, 0.03, 0, 0.45, 0.04,
      // Left eye landmark loop
      -0.16, 0.12, 0.05, -0.06, 0.14, 0.05,
      -0.06, 0.14, 0.05, -0.1, 0.08, 0.05,
      -0.1, 0.08, 0.05, -0.16, 0.12, 0.05,
      // Right eye landmark loop
      0.06, 0.14, 0.05, 0.16, 0.12, 0.05,
      0.16, 0.12, 0.05, 0.1, 0.08, 0.05,
      0.1, 0.08, 0.05, 0.06, 0.14, 0.05,
      // Nose ridge
      0, 0.15, 0.06, 0, -0.08, 0.08,
      0, -0.08, 0.08, -0.06, -0.12, 0.06,
      0, -0.08, 0.08, 0.06, -0.12, 0.06,
      // Mouth contour
      -0.1, -0.26, 0.04, 0, -0.22, 0.05,
      0, -0.22, 0.05, 0.1, -0.26, 0.04,
      0.1, -0.26, 0.04, 0, -0.32, 0.04,
      0, -0.32, 0.04, -0.1, -0.26, 0.04,
    ];
    faceGeo.setAttribute(
      'position',
      new THREE.Float32BufferAttribute(facePoints, 3)
    );
    const faceMat = new THREE.LineBasicMaterial({
      color: 0x66f6ff,
      transparent: true,
      opacity: 0.85,
    });
    this.faceWireframe = new THREE.LineSegments(faceGeo, faceMat);
    this.faceWireframe.position.set(0, 1.45, 0.05);
    this.group.add(this.faceWireframe);

    // 4. Sweeping Laser Scan Line (GradCAM Inspection simulation)
    const scanGeo = new THREE.BufferGeometry().setFromPoints([
      new THREE.Vector3(-frameWidth * 0.46, 0, 0.06),
      new THREE.Vector3(frameWidth * 0.46, 0, 0.06),
    ]);
    const scanMat = new THREE.LineBasicMaterial({
      color: 0x00ffff,
      transparent: true,
      opacity: 0.95,
      linewidth: 3,
    });
    this.scanLine = new THREE.Line(scanGeo, scanMat);
    this.scanLine.position.set(0, 1.45, 0);
    this.group.add(this.scanLine);

    // 5. Floating Interactive Beacon
    const beaconGeo = new THREE.OctahedronGeometry(0.18, 0);
    const beaconMat = new THREE.MeshStandardMaterial({
      color: 0x00f0ff,
      emissive: 0x00f0ff,
      emissiveIntensity: 0.8,
      roughness: 0.1,
    });
    this.beacon = new THREE.Mesh(beaconGeo, beaconMat);
    this.beacon.position.set(0, 2.45, 0);
    this.group.add(this.beacon);

    // Set interactive userData on primary pickable meshes
    this.pedestal.userData = { isStation: true, stationId: 'spatial' };
    backplane.userData = { isStation: true, stationId: 'spatial' };
    this.beacon.userData = { isStation: true, stationId: 'spatial' };
  }

  public setHovered(hovered: boolean) {
    this.isHovered = hovered;
  }

  public setSelected(selected: boolean) {
    this.isSelected = selected;
  }

  public update(delta: number, elapsed: number) {
    // Laser scan animation
    this.scanY += delta * 0.8 * this.scanDirection;
    if (this.scanY > 0.55) {
      this.scanY = 0.55;
      this.scanDirection = -1;
    } else if (this.scanY < -0.55) {
      this.scanY = -0.55;
      this.scanDirection = 1;
    }
    this.scanLine.position.y = 1.45 + this.scanY;

    // Beacon rotation and float
    this.beacon.rotation.y += delta * 1.5;
    this.beacon.rotation.x += delta * 0.8;
    this.beacon.position.y = 2.45 + Math.sin(elapsed * 2.5) * 0.06;

    // Frame tilt towards center slightly
    const targetY = this.isHovered || this.isSelected ? 0.08 : 0.0;
    this.frameBorder.rotation.y = THREE.MathUtils.lerp(
      this.frameBorder.rotation.y,
      0.35 + (this.isHovered ? 0.1 : 0),
      0.1
    );

    // Scale feedback
    const targetScale = this.isSelected ? 1.08 : this.isHovered ? 1.04 : 1.0;
    this.frameBorder.scale.lerp(new THREE.Vector3(targetScale, targetScale, targetScale), 0.1);

    // Pedestal glow
    const pedMat = this.pedestal.material as THREE.MeshStandardMaterial;
    pedMat.emissiveIntensity = this.isSelected
      ? 0.8 + Math.sin(elapsed * 4.0) * 0.2
      : this.isHovered
      ? 0.6
      : 0.25;
  }

  public dispose() {
    this.pedestal.geometry.dispose();
    (this.pedestal.material as THREE.Material).dispose();
    this.frameBorder.geometry.dispose();
    (this.frameBorder.material as THREE.Material).dispose();
    this.faceWireframe.geometry.dispose();
    (this.faceWireframe.material as THREE.Material).dispose();
    this.scanLine.geometry.dispose();
    (this.scanLine.material as THREE.Material).dispose();
    this.beacon.geometry.dispose();
    (this.beacon.material as THREE.Material).dispose();
  }
}
