import { useEffect, useRef, useCallback } from "react";
import { useQuery, useClient } from "urql";
import { useTrieDataStore } from "../store/trieDataStore";
import { useFilterStore } from "../store/filterStore";
import { METADATA_QUERY, NODES_BY_DEPTH_QUERY } from "../graphql/queries";
import { toRenderNode } from "../types/trie";

/**
 * Progressive loading hook:
 * 1. Fetch metadata (tiny)
 * 2. Fetch depth 0-2 immediately
 * 3. As visibleDepth increases (from LOD), fetch deeper levels
 */
export function useTrieData(visibleDepth: number) {
  const client = useClient();
  const {
    setMetadata,
    addNodes,
    addEdges,
    markDepthLoaded,
    loadedDepths,
    setLoading,
    setError,
  } = useTrieDataStore();

  const setMaxDepthLimit = useFilterStore((s) => s.setMaxDepthLimit);
  const loadingRef = useRef(new Set<number>());

  // Phase 1: Fetch metadata
  const [metaResult] = useQuery({ query: METADATA_QUERY });

  useEffect(() => {
    if (metaResult.data?.metadata) {
      setMetadata(metaResult.data.metadata);
      setMaxDepthLimit(metaResult.data.metadata.maxDepth);
    }
    if (metaResult.error) {
      setError(metaResult.error.message);
    }
  }, [metaResult.data, metaResult.error, setMetadata, setMaxDepthLimit, setError]);

  // Fetch a depth range
  const fetchDepthRange = useCallback(
    async (minDepth: number, maxDepth: number) => {
      // Check if already loaded or loading
      for (let d = minDepth; d <= maxDepth; d++) {
        if (loadedDepths.has(d) || loadingRef.current.has(d)) return;
      }

      for (let d = minDepth; d <= maxDepth; d++) {
        loadingRef.current.add(d);
      }

      setLoading(true);
      try {
        const result = await client
          .query(NODES_BY_DEPTH_QUERY, {
            minDepth,
            maxDepth,
            limit: 10000,
          })
          .toPromise();

        if (result.error) {
          setError(result.error.message);
          return;
        }

        if (result.data) {
          const nodes = result.data.nodesByDepth.nodes.map(toRenderNode);
          addNodes(nodes);
          addEdges(result.data.edgesByDepth);
          for (let d = minDepth; d <= maxDepth; d++) {
            markDepthLoaded(d);
          }
        }
      } finally {
        for (let d = minDepth; d <= maxDepth; d++) {
          loadingRef.current.delete(d);
        }
        setLoading(false);
      }
    },
    [client, loadedDepths, addNodes, addEdges, markDepthLoaded, setLoading, setError],
  );

  // Phase 2: Initial load (depth 0-2)
  useEffect(() => {
    if (metaResult.data?.metadata && !loadedDepths.has(0)) {
      fetchDepthRange(0, 2);
    }
  }, [metaResult.data, loadedDepths, fetchDepthRange]);

  // Phase 3: Progressive loading based on visible depth
  useEffect(() => {
    const maxLoaded = Math.max(0, ...loadedDepths);
    if (visibleDepth > maxLoaded && metaResult.data?.metadata) {
      const nextMin = maxLoaded + 1;
      const nextMax = Math.min(visibleDepth, metaResult.data.metadata.maxDepth);
      if (nextMin <= nextMax) {
        fetchDepthRange(nextMin, nextMax);
      }
    }
  }, [visibleDepth, loadedDepths, fetchDepthRange, metaResult.data]);
}
