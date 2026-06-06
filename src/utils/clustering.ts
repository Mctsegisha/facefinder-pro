import { cosineSimilarity } from "../services/geminiService";

export interface ClusterNode {
  url: string;
  embedding: number[];
  faceBox?: { x: number; y: number; w: number; h: number };
  category?: string;
  timestamp?: string;
  index: number;
}

export interface Cluster {
  id: number;
  label: string;
  nodes: ClusterNode[];
  representative: ClusterNode;
}

/**
 * Performs DBSCAN clustering on indexed photos using vector cosine similarities
 * 
 * @param nodes List of photos with embeddings
 * @param similarityThreshold Threshold score (e.g. 0.80) to consider faces as the same person
 * @returns Array of grouped face clusters sorted by size
 */
export function performDBSCANClustering(
  nodes: { url: string; embedding: number[] | null; faceBox?: any; category?: string; timestamp?: string }[],
  similarityThreshold: number = 0.80
): Cluster[] {
  // Filter out any nodes that do not contain valid vector embeddings
  const validNodes: ClusterNode[] = nodes
    .filter(n => n.embedding !== null)
    .map((n, i) => ({
      url: n.url,
      embedding: n.embedding!,
      faceBox: n.faceBox,
      category: n.category,
      timestamp: n.timestamp,
      index: i
    }));

  const n = validNodes.length;
  if (n === 0) return [];

  const visited = new Array(n).fill(false);
  const clusterIds = new Array(n).fill(-1); // -1 signifies noise / unassigned
  let currentClusterId = 0;

  // Find neighbor indices based on cosine similarity limit
  function getNeighbors(nodeIdx: number): number[] {
    const neighbors: number[] = [];
    const targetVector = validNodes[nodeIdx].embedding;
    for (let i = 0; i < n; i++) {
      if (i === nodeIdx) continue;
      const sim = cosineSimilarity(targetVector, validNodes[i].embedding);
      if (sim >= similarityThreshold) {
        neighbors.push(i);
      }
    }
    return neighbors;
  }

  // DBSCAN Expansion Loop (MinPts = 1 to capture all distinct individuals)
  for (let i = 0; i < n; i++) {
    if (visited[i]) continue;
    visited[i] = true;

    const neighbors = getNeighbors(i);
    
    // Assign starting node to the current cluster
    clusterIds[i] = currentClusterId;
    
    const queue = [...neighbors];
    for (let q = 0; q < queue.length; q++) {
      const neighborIdx = queue[q];
      if (!visited[neighborIdx]) {
        visited[neighborIdx] = true;
        const nextNeighbors = getNeighbors(neighborIdx);
        
        // Append unique index points to queue
        for (const idx of nextNeighbors) {
          if (!queue.includes(idx)) {
            queue.push(idx);
          }
        }
      }
      if (clusterIds[neighborIdx] === -1) {
        clusterIds[neighborIdx] = currentClusterId;
      }
    }
    
    currentClusterId++;
  }

  // Partition elements based on grouped IDs
  const clustersMap: { [id: number]: ClusterNode[] } = {};
  for (let i = 0; i < n; i++) {
    const id = clusterIds[i];
    if (id === -1) continue;
    if (!clustersMap[id]) {
      clustersMap[id] = [];
    }
    clustersMap[id].push(validNodes[i]);
  }

  // Map into structured Cluster schemas
  const clusters: Cluster[] = Object.entries(clustersMap).map(([idStr, nodesList]) => {
    const id = parseInt(idStr, 10);
    // Use the first added node as the representative thumbnail
    const representative = nodesList[0];
    return {
      id,
      label: `Person #${id + 1}`,
      nodes: nodesList,
      representative
    };
  });

  // Sort clusters by cluster size (descending) so prominent people show up first
  return clusters.sort((a, b) => b.nodes.length - a.nodes.length);
}
