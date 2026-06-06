import express from "express";
import { createServer as createViteServer } from "vite";
import path from "path";
import multer from "multer";
import { v4 as uuidv4 } from "uuid";
import fs from "fs";
import dotenv from "dotenv";
import { createClient } from "@supabase/supabase-js";

// Load environment variables
dotenv.config();

// Simple cosine similarity helper for backend local search fallback
function cosineSimilarity(vecA: number[], vecB: number[]) {
  if (vecA.length !== vecB.length) return 0;
  let dotProduct = 0;
  let normA = 0;
  let normB = 0;
  for (let i = 0; i < vecA.length; i++) {
    dotProduct += vecA[i] * vecB[i];
    normA += vecA[i] * vecA[i];
    normB += vecB[i] * vecB[i];
  }
  if (normA === 0 || normB === 0) return 0;
  return dotProduct / (Math.sqrt(normA) * Math.sqrt(normB));
}

async function startServer() {
  const app = express();
  const PORT = 3000;

  // Supabase Configuration
  const supabaseUrl = process.env.SUPABASE_URL || "";
  const supabaseServiceKey = process.env.SUPABASE_SERVICE_ROLE_KEY || process.env.SUPABASE_ANON_KEY || "";
  const useSupabase = !!(supabaseUrl && supabaseServiceKey);

  let supabase: ReturnType<typeof createClient> | null = null;
  if (useSupabase) {
    console.log("Supabase credentials detected. Enabling Cloud Database & Storage Engine.");
    supabase = createClient(supabaseUrl, supabaseServiceKey);
  } else {
    console.log("No Supabase configuration detected. Using Local Disk & In-Memory Fallback Engine.");
  }

  // Ensure local uploads directory exists (for fallback mode)
  const uploadsDir = path.join(process.cwd(), "public", "uploads");
  if (!fs.existsSync(uploadsDir)) {
    fs.mkdirSync(uploadsDir, { recursive: true });
  }

  // Configure Multer storage depending on active mode
  const upload = useSupabase
    ? multer({ storage: multer.memoryStorage() }) // Stream buffers directly to cloud storage
    : multer({
        storage: multer.diskStorage({
          destination: (req, file, cb) => {
            cb(null, uploadsDir);
          },
          filename: (req, file, cb) => {
            const uniqueName = `${uuidv4()}${path.extname(file.originalname)}`;
            cb(null, uniqueName);
          },
        }),
      });

  app.use(express.json({ limit: "50mb" })); // Support large base64 image transmissions if needed
  app.use(express.urlencoded({ extended: true, limit: "50mb" }));

  // 1. Photo Upload Endpoint
  app.post("/api/upload", upload.array("photos"), async (req, res) => {
    try {
      const files = req.files as Express.Multer.File[];
      if (!files || files.length === 0) {
        return res.status(400).json({ error: "No photo files uploaded" });
      }

      const photoUrls: string[] = [];

      if (useSupabase && supabase) {
        // Upload images to Supabase Storage Bucket ('photos')
        for (const file of files) {
          const uniqueName = `${uuidv4()}${path.extname(file.originalname)}`;
          const { error } = await supabase.storage
            .from("photos")
            .upload(uniqueName, file.buffer, {
              contentType: file.mimetype,
              upsert: true,
            });

          if (error) {
            console.error("Supabase Storage Upload Error:", error);
            return res.status(500).json({ error: "Failed to upload assets to Supabase Storage." });
          }

          const { data: { publicUrl } } = supabase.storage
            .from("photos")
            .getPublicUrl(uniqueName);

          photoUrls.push(publicUrl);
        }
      } else {
        // Fallback: save to local disk
        for (const file of files) {
          photoUrls.push(`/uploads/${file.filename}`);
        }
      }

      res.json({ success: true, urls: photoUrls });
    } catch (err) {
      console.error("Upload Endpoint Error:", err);
      res.status(500).json({ error: "Server upload processing error" });
    }
  });

  // Local vector storage fallback dictionary
  const vectorStore: {
    [url: string]: {
      embedding: number[];
      faceBox?: any;
      category?: string;
      timestamp?: string;
    };
  } = {};

  // 2. Vector Index Endpoint (Register Photo Metadata & Embedding)
  app.post("/api/index", async (req, res) => {
    const { url, embedding, faceBox, category, timestamp } = req.body;

    if (!url || !embedding || !Array.isArray(embedding)) {
      return res.status(400).json({ error: "Missing required indexing properties: url or embedding" });
    }

    if (useSupabase && supabase) {
      try {
        const { error } = await supabase
          .from("photos")
          .insert({
            url,
            category: category || "Candid Social Interaction",
            face_box: faceBox || null,
            timestamp: timestamp || new Date().toISOString().slice(11, 19),
            embedding: embedding, // pgvector handles JS arrays natively
          });

        if (error) {
          console.error("Supabase Database Insert Error:", error);
          return res.status(500).json({ error: "Failed to write face index to cloud database." });
        }
      } catch (err) {
        console.error("Supabase Index Exception:", err);
        return res.status(500).json({ error: "Supabase indexing exception" });
      }
    } else {
      // Local fallback
      vectorStore[url] = {
        embedding,
        faceBox,
        category: category || "Candid Social Interaction",
        timestamp: timestamp || new Date().toISOString().slice(11, 19),
      };
    }

    res.json({ success: true });
  });

  // 3. Get All Indexed Photos List
  app.get("/api/photos", async (req, res) => {
    if (useSupabase && supabase) {
      try {
        const { data, error } = await supabase
          .from("photos")
          .select("url, embedding, face_box, category, timestamp")
          .order("created_at", { ascending: false });

        if (error) {
          console.error("Supabase List Query Error:", error);
          return res.status(500).json({ error: "Failed to fetch photo indexes." });
        }

        const formatted = data.map((item: any) => ({
          url: item.url,
          embedding: item.embedding ? Array.from(item.embedding) : null,
          faceBox: item.face_box,
          category: item.category,
          timestamp: item.timestamp,
        }));

        res.json({ photos: formatted });
      } catch (err) {
        console.error("Supabase List Exception:", err);
        res.status(500).json({ error: "Cloud query exception" });
      }
    } else {
      // Local Fallback: read uploads directory
      fs.readdir(uploadsDir, (err, files) => {
        if (err) {
          return res.status(500).json({ error: "Failed to list local photos" });
        }
        const urls = files
          .filter((file) => /\.(jpg|jpeg|png|webp)$/i.test(file))
          .map((file) => `/uploads/${file}`);

        const photosWithEmbeddings = urls.map((url) => {
          const data = vectorStore[url];
          return {
            url,
            embedding: data ? data.embedding : null,
            faceBox: data ? data.faceBox : null,
            category: data ? data.category : "Candid Social Interaction",
            timestamp: data ? data.timestamp : null,
          };
        });

        res.json({ photos: photosWithEmbeddings });
      });
    }
  });

  // 4. Server-Side Vector Search Endpoint
  app.post("/api/search", async (req, res) => {
    const { embedding, category = "All" } = req.body;

    if (!embedding || !Array.isArray(embedding)) {
      return res.status(400).json({ error: "Invalid or missing search embedding parameter" });
    }

    if (useSupabase && supabase) {
      try {
        // Run pgvector Cosine similarity query using PostgreSQL RPC function 'match_photos'
        const { data, error } = await supabase.rpc("match_photos", {
          query_embedding: embedding,
          match_threshold: 0.60,
          match_count: 50,
          filter_category: category,
        });

        if (error) {
          console.error("Supabase Search Query Error:", error);
          return res.status(500).json({ error: "Cloud database similarity query failed." });
        }

        const matches = data.map((item: any) => ({
          url: item.url,
          score: item.similarity,
          faceBox: item.face_box,
          category: item.category,
          timestamp: item.timestamp,
        }));

        res.json({ success: true, results: matches });
      } catch (err) {
        console.error("Supabase Search Exception:", err);
        res.status(500).json({ error: "Cloud similarity search exception" });
      }
    } else {
      // Local Fallback: perform cosine similarity in-memory
      const matches = Object.entries(vectorStore)
        .map(([url, data]) => {
          const score = cosineSimilarity(embedding, data.embedding);
          return {
            url,
            score,
            faceBox: data.faceBox,
            category: data.category,
            timestamp: data.timestamp,
          };
        })
        .filter((match) => match.score > 0.60 && (category === "All" || match.category === category))
        .sort((a, b) => b.score - a.score);

      res.json({ success: true, results: matches });
    }
  });

  // Vite middleware for development
  if (process.env.NODE_ENV !== "production") {
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: "spa",
    });
    app.use(vite.middlewares);
  } else {
    const distPath = path.join(process.cwd(), "dist");
    app.use(express.static(distPath));
    app.get("*", (req, res) => {
      res.sendFile(path.join(distPath, "index.html"));
    });
  }

  app.listen(PORT, "0.0.0.0", () => {
    console.log(`Server running on http://localhost:${PORT}`);
  });
}

startServer();
