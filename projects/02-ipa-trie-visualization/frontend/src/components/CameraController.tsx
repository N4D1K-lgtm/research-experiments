import { useRef } from "react";
import { useFrame, useThree } from "@react-three/fiber";
import { OrbitControls } from "@react-three/drei";

interface Props {
  onDistanceChange: (distance: number) => void;
}

export function CameraController({ onDistanceChange }: Props) {
  const lastDistance = useRef(0);
  const { camera } = useThree();

  useFrame(() => {
    const d = camera.position.length();
    // Only notify on significant change
    if (Math.abs(d - lastDistance.current) > 1) {
      lastDistance.current = d;
      onDistanceChange(d);
    }
  });

  return (
    <OrbitControls
      enableDamping
      dampingFactor={0.06}
      minDistance={10}
      maxDistance={800}
    />
  );
}
