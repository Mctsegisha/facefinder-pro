/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import { generateFallbackEmbedding } from "../services/geminiService";

export interface SimulatedPhoto {
  url: string;
  embedding: number[];
  category: string;
  faceBox?: { x: number; y: number; w: number; h: number };
  timestamp: string;
}

const CONST_CATEGORIES = [
  "Auditorium Presentation",
  "Attendee Networking",
  "Panel Q&A Panelist",
  "Keynote Speaker Highlight",
  "Interactive Demo Arena",
  "Backstage Interview Session",
  "Candid Social Interaction"
];

const UNSPLASH_IDS: { [category: string]: string[] } = {
  "Auditorium Presentation": [
    "photo-1540575467063-178a50c2df87",
    "photo-1475721027785-f74eccf877e2",
    "photo-1511578314322-379afb476865",
    "photo-1505373877841-8d25f7d46678",
    "photo-1492684223066-81342ee5ff30"
  ],
  "Attendee Networking": [
    "photo-1515187029135-18ee286d815b",
    "photo-1528605248644-14dd04022da1",
    "photo-1511795409834-ef04bbd61622",
    "photo-1517048676732-d65bc937f952",
    "photo-1519389950473-47ba0277781c"
  ],
  "Panel Q&A Panelist": [
    "photo-1531482615713-2afd69097998",
    "photo-1556761175-4b46a572b786",
    "photo-1434030216411-0b793f4b4173",
    "photo-1522071820081-009f0129c71c",
    "photo-1522202176988-66273c2fd55f"
  ],
  "Keynote Speaker Highlight": [
    "photo-1506157786151-b8491531f063",
    "photo-1516280440614-37939bbacd6a",
    "photo-1461280360983-bd93eaa50516",
    "photo-1568992687947-868a62a9f521",
    "photo-1558224494-ef8b2450d994"
  ],
  "Interactive Demo Arena": [
    "photo-1581092921461-eab62e97a780",
    "photo-1535378917042-10a22c95931a",
    "photo-1558494949-ef010cbdcc31",
    "photo-1563986768609-322da13575f3",
    "photo-1460925895917-afdab827c52f"
  ],
  "Backstage Interview Session": [
    "photo-1488590528505-98d2b5aba04b",
    "photo-1573164713988-8665fc963095",
    "photo-1516321318423-f06f85e504b3",
    "photo-1542744094-3a31f103e35f",
    "photo-1507238691740-187a5b1d37b8"
  ],
  "Candid Social Interaction": [
    "photo-1543269865-cbf427effbad",
    "photo-1517486808906-6ca8b3f04846",
    "photo-1523240795612-9a054b0db644",
    "photo-1543269664-76bc3997d9ea",
    "photo-1529156069898-49953e39b3ac"
  ]
};

// Generate repeatable random values based on a seed
class SeededRandom {
  private seed: number;

  constructor(seed: number) {
    this.seed = seed;
  }

  // Linear Congruential Generator
  next(): number {
    this.seed = (this.seed * 1664525 + 1013904223) % 4294967296;
    return this.seed / 4294967296;
  }

  // Random within range
  range(min: number, max: number): number {
    return min + this.next() * (max - min);
  }

  // Random element of array
  choice<T>(arr: T[]): T {
    const idx = Math.floor(this.next() * arr.length);
    return arr[idx];
  }
}

/**
 * Generates thousands of high-fidelity photo metadata instances dynamically
 * with deterministic vector embeddings (256-d) and Unsplash source URLs.
 * 
 * @param count Number of event photos to generate
 * @returns Array of simulated photo nodes
 */
export function generateEventPhotos(count: number): SimulatedPhoto[] {
  const photosList: SimulatedPhoto[] = [];
  const rand = new SeededRandom(42); // Consistent seed for repeatable demo data

  for (let i = 0; i < count; i++) {
    // Pick category
    const category = rand.choice(CONST_CATEGORIES);
    const ids = UNSPLASH_IDS[category];
    const baseId = rand.choice(ids);
    
    // Create a distinct image URL by appending a crop & dynamic query parameters that make them unique images to the browser or mock indices
    const qualityParam = Math.floor(rand.range(60, 95));
    const widthParam = Math.floor(rand.range(300, 500));
    // Unsplash allows passing 'sig' parameter to force uniqueness or randomized versions
    const url = `https://images.unsplash.com/${baseId}?auto=format&fit=crop&q=${qualityParam}&w=${widthParam}&sig=${i}`;

    // Compute a pseudo-random deterministic vector of 128 dimensions
    // We base it on seed so it is stable and fast to compute
    const embedding: number[] = new Array(128).fill(0);
    let sumSqr = 0;
    for (let d = 0; d < 128; d++) {
      // Use trigonometric mixes to compute a unique deterministic coordinates spread
      const val = Math.sin(i * 3.7 + d) * Math.cos(d * 11.3 + i * 0.17);
      embedding[d] = val;
      sumSqr += val * val;
    }
    
    // Normalize to unit length (L2 norm) so cosine similarity search matches standard expectations perfectly
    const mag = Math.sqrt(sumSqr);
    if (mag > 0) {
      for (let d = 0; d < 128; d++) {
        embedding[d] /= mag;
      }
    }

    // Interactive face box layout coordinates
    const faceBox = {
      x: Math.floor(rand.range(15, 45)),
      y: Math.floor(rand.range(15, 40)),
      w: Math.floor(rand.range(25, 45)),
      h: Math.floor(rand.range(30, 50))
    };

    // Human repeatable timestamps
    const date = new Date(Date.now() - Math.floor(rand.range(0, 1) * 3 * 24 * 3600 * 1000));
    const timestamp = date.toISOString().slice(11, 19);

    photosList.push({
      url,
      embedding,
      category,
      faceBox,
      timestamp
    });
  }

  return photosList;
}
