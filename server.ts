import express from "express";
import { createServer as createViteServer } from "vite";
import path from "path";
import multer from "multer";
import { v4 as uuidv4 } from "uuid";
import fs from "fs";

async function startServer() {
  const app = express();
  const PORT = 3000;

  // Ensure uploads directory exists
  const uploadsDir = path.join(process.cwd(), "public", "uploads");
  if (!fs.existsSync(uploadsDir)) {
    fs.mkdirSync(uploadsDir, { recursive: true });
  }

  // Configure Multer for file uploads
  const storage = multer.diskStorage({
    destination: (req, file, cb) => {
      cb(null, uploadsDir);
    },
    filename: (req, file, cb) => {
      const uniqueName = `${uuidv4()}${path.extname(file.originalname)}`;
      cb(null, uniqueName);
    },
  });

  const upload = multer({ storage });

  app.use(express.json());

  // API Routes
  app.post("/api/upload", upload.array("photos"), (req, res) => {
    const files = req.files as Express.Multer.File[];
    const photoUrls = files.map((file) => `/uploads/${file.filename}`);
    res.json({ success: true, urls: photoUrls });
  });

  // Store embeddings in memory for this demo
  const vectorStore: { [url: string]: number[] } = {};

  app.post("/api/index", (req, res) => {
    const { url, embedding } = req.body;
    vectorStore[url] = embedding;
    res.json({ success: true });
  });

  app.get("/api/photos", (req, res) => {
    fs.readdir(uploadsDir, (err, files) => {
      if (err) {
        return res.status(500).json({ error: "Failed to list photos" });
      }
      const urls = files
        .filter((file) => /\.(jpg|jpeg|png|webp)$/i.test(file))
        .map((file) => `/uploads/${file}`);
      
      const photosWithEmbeddings = urls.map(url => ({
        url,
        embedding: vectorStore[url] || null
      }));

      res.json({ photos: photosWithEmbeddings });
    });
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
