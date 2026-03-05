import { useState, useCallback } from "react";
import { computeLOD, type LODState } from "../utils/lod";

export function useLOD(maxDepth: number) {
  const [lod, setLOD] = useState<LODState>({ visibleMaxDepth: 5, fadeDepth: 5 });

  const updateLOD = useCallback(
    (cameraDistance: number) => {
      const newLOD = computeLOD(cameraDistance, maxDepth);
      setLOD((prev) => {
        if (
          prev.visibleMaxDepth === newLOD.visibleMaxDepth &&
          prev.fadeDepth === newLOD.fadeDepth
        ) {
          return prev;
        }
        return newLOD;
      });
    },
    [maxDepth],
  );

  return { lod, updateLOD };
}
