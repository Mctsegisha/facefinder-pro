# FaceFinder Pro — Biometric Event Photo Discovery Platform

A high-performance facial recognition and event photo discovery system. Developed by Tsegab, this application allows photographers to upload high-resolution event folders, automatically groups photos by identified individuals using unsupervised clustering, and lets attendees find their photos instantaneously using a selfie upload or live webcam video tracking.

---

## Core Technologies

*   **Frontend:** React 19, Vite 6, Tailwind CSS v4, Motion, Lucide Icons.
*   **Backend:** Node.js, Express, TypeScript, Multer.
*   **Database & Vector Store:** PostgreSQL + `pgvector` (via Supabase).
*   **Biometrics & AI:** Client-side `face-api.js` (SSD MobileNet V1 face detector + Face Landmarks + 128-dimensional face embedding extraction).
*   **Object Storage:** Supabase Storage (100% free tier-compatible).

---

## Advanced Features

1.  **Browser-Based Face Extraction (100% Free):** Biometric vectors and facial coordinates are extracted locally in the user's browser, eliminating CPU loads and third-party AI API fees.
2.  **Unsupervised DBSCAN Face Clustering:** Automatically parses all faces in the database and clusters them into distinct person profiles. Browse matching photos grouped by detected people without uploading a selfie.
3.  **Real-Time Video Matching:** Toggling Live Tracking on the Attendee webcam continuously processes the camera stream at 1.2s intervals, dynamically fading in matching event photos as you move.
4.  **SVG Bounding Box Inspector:** Clicking any photo overlays bounding boxes on all detected faces. Hovering over a face displays its 128-dimensional vector embedding projected as a dynamic bar graph.
5.  **Interactive Vector Manifold Map:** Visualizes the entire index as a projected 2D constellation network. Nodes share connection lines based on similarity thresholds and can be clicked to inspect photo details.
6.  **Dual-Mode Operation:** Starts in **Local Fallback Mode** (local uploads and in-memory caches) out-of-the-box, then seamlessly upgrades to **Cloud Database Mode** when Supabase keys are provided in the environment.

---

## Local Setup

### 1. Install Dependencies
```bash
npm install
```

### 2. Configure Environment Variables
Create a `.env` file in the root directory:
```env
SUPABASE_URL="your-supabase-project-url"
SUPABASE_ANON_KEY="your-supabase-anon-key"
SUPABASE_SERVICE_ROLE_KEY="your-supabase-service-role-key"
APP_URL="http://localhost:3000"
```
*(Leave Supabase variables empty if you want to run the project locally in **Local Fallback Mode** without a database setup).*

### 3. Initialize Database (For Supabase Cloud Mode)
1.  Paste the contents of [supabase_schema.sql](supabase_schema.sql) in your Supabase SQL Editor and run it.
2.  Create a public bucket named `photos` in your Supabase Storage console.

### 4. Run the Application
```bash
npm run dev
```
Open **`http://localhost:3000`** in your browser.

---

## Deployment (100% Free)

*   **Database & Storage:** Hosted on [Supabase](https://supabase.com) free tier.
*   **Web Server & API:** Easily deployable on [Render.com](https://render.com) free tier (Build command: `npm run build`, Start command: `npm start`) or [Vercel](https://vercel.com) using the included [vercel.json](vercel.json).
