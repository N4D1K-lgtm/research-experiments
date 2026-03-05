import { gql } from "urql";

export const METADATA_QUERY = gql`
  query Metadata {
    metadata {
      languages
      nodeCount
      edgeCount
      maxDepth
      totalWords
      terminalNodes
      phonemeInventory
      onsetInventory
      codaInventory
      motifs {
        sequence
        count
        label
      }
    }
  }
`;

const NODE_FRAGMENT = `
  id
  phoneme
  depth
  parentId
  totalCount
  position { x y z }
  color
  hsl { h s l }
  phonologicalPosition
  isTerminal
  childCount
  weight
  transitionProbs { phoneme probability }
  allophones
  terminalCounts { language count }
  words { language words }
`;

export const NODES_BY_DEPTH_QUERY = gql`
  query NodesByDepth($minDepth: Int!, $maxDepth: Int!, $offset: Int, $limit: Int) {
    nodesByDepth(minDepth: $minDepth, maxDepth: $maxDepth, offset: $offset, limit: $limit) {
      nodes {
        ${NODE_FRAGMENT}
      }
      totalCount
      hasMore
    }
    edgesByDepth(minDepth: $minDepth, maxDepth: $maxDepth) {
      source
      target
    }
  }
`;

export const NODE_DETAIL_QUERY = gql`
  query NodeDetail($id: Int!) {
    node(id: $id) {
      ${NODE_FRAGMENT}
      children {
        id
        phoneme
        totalCount
        phonologicalPosition
        isTerminal
      }
      parent {
        id
        phoneme
        transitionProbs { phoneme probability }
      }
    }
  }
`;

export const SEARCH_QUERY = gql`
  query Search($phoneme: String!, $limit: Int) {
    search(phoneme: $phoneme, limit: $limit) {
      nodes {
        ${NODE_FRAGMENT}
      }
      totalMatches
    }
  }
`;

export const DEPTH_STATS_QUERY = gql`
  query DepthStats {
    depthStats {
      depth
      nodes
      terminals
      avgBranch
      avgEntropy
      maxEntropy
    }
  }
`;

export const LANGUAGES_QUERY = gql`
  query Languages {
    languages {
      code
      name
      family
      typology
      iso6393
    }
  }
`;

export const CROSS_LINGUISTIC_STATS_QUERY = gql`
  query CrossLinguisticStats {
    crossLinguisticStats {
      language
      stats {
        depth
        nodes
        terminals
        avgBranch
        avgEntropy
      }
    }
  }
`;
