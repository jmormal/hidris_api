import express from "express";

const app = express();
app.use(express.json());
app.use((req, res, next) => {
  res.header("Access-Control-Allow-Origin", "*");
  res.header("Access-Control-Allow-Headers", "Content-Type, Authorization");
  res.header("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
  if (req.method === "OPTIONS") return res.sendStatus(204);
  next();
});
// Inside the cluster, services talk via their k8s Service name.
// This never leaves the cluster — no nip.io needed here.
const AUTH_URL = process.env.AUTH_URL ?? "http://auth:80";

app.get("/health", (_req, res) => res.json({ status: "ok", service: "api" }));

// GET /items  — public
app.get("/items", (_req, res) => {
  res.json([
    { id: 1, name: "Widget A" },
    { id: 2, name: "Wi:Widget j" },
    { id: 3, name: "Widget C" },
  ]);
});

// GET /me  — protected, validates Bearer token via auth service
app.get("/me", async (req, res) => {
  const token = req.headers.authorization?.replace("Bearer ", "");
  if (!token) return res.status(401).json({ error: "no token" });

  try {
    const authRes = await fetch(`${AUTH_URL}/validate`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ token }),
    });
    const data = await authRes.json();
    if (!data.valid) return res.status(401).json({ error: "invalid token" });
    res.json({ user: data.user });
  } catch (err) {
    console.error("auth service error:", err.message);
    res.status(502).json({ error: "auth service unavailable" });
  }
});

const PORT = process.env.PORT ?? 8080;
app.listen(PORT, () => console.log(`api listening on :${PORT}`));
