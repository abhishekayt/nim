import express, { type Express } from "express";
import cors from "cors";
import messagesRouter from "./routes/messages";
import adminRouter from "./routes/admin";
// eslint-disable-next-line @typescript-eslint/ban-ts-comment
// @ts-expect-error - esbuild loads .html as text
import dashboardHtml from "./dashboard.html";

export function createApp(): Express {
  const app = express();
  app.use(cors());
  app.use(express.json({ limit: "20mb" }));
  app.use(express.urlencoded({ extended: true }));

  app.get("/api/healthz", (_req, res) => res.json({ status: "ok" }));
  app.use(messagesRouter);
  app.use(adminRouter);
  app.get("/", (_req, res) => res.type("html").send(dashboardHtml as string));

  return app;
}

export function startServer(port: number): Promise<void> {
  return new Promise((resolve, reject) => {
    const app = createApp();
    const server = app.listen(port, (err?: Error) => {
      if (err) reject(err);
      else resolve();
    });
    server.on("error", reject);
  });
}
