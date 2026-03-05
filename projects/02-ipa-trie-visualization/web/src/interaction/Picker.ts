import * as THREE from "three";
import type { NodeRenderer } from "../rendering/NodeRenderer";
import type { TrieNodeData } from "../types";

export class Picker {
  private raycaster = new THREE.Raycaster();
  private mouse = new THREE.Vector2();
  private camera: THREE.Camera;
  private nodeRenderer: NodeRenderer;
  private onHover: (node: TrieNodeData | null, x: number, y: number) => void;
  private lastInstanceId = -1;

  constructor(
    camera: THREE.Camera,
    nodeRenderer: NodeRenderer,
    onHover: (node: TrieNodeData | null, x: number, y: number) => void,
  ) {
    this.camera = camera;
    this.nodeRenderer = nodeRenderer;
    this.onHover = onHover;

    window.addEventListener("mousemove", this.handleMove.bind(this));
  }

  private handleMove(e: MouseEvent): void {
    this.mouse.x = (e.clientX / window.innerWidth) * 2 - 1;
    this.mouse.y = -(e.clientY / window.innerHeight) * 2 + 1;

    this.raycaster.setFromCamera(this.mouse, this.camera);
    const intersects = this.raycaster.intersectObject(this.nodeRenderer.mesh);

    if (intersects.length > 0) {
      const instanceId = intersects[0].instanceId;
      if (instanceId != null && instanceId !== this.lastInstanceId) {
        this.lastInstanceId = instanceId;
        const node = this.nodeRenderer.getNodeAtInstance(instanceId);
        this.onHover(node, e.clientX, e.clientY);
      }
    } else {
      if (this.lastInstanceId !== -1) {
        this.lastInstanceId = -1;
        this.onHover(null, 0, 0);
      }
    }
  }
}
