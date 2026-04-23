import express, { type Express } from "express";
import path from "node:path";
import { fileURLToPath } from "node:url";
import cors from "cors";
import pinoHttp from "pino-http";
import router from "./routes";
import messagesRouter from "./routes/messages";
import adminRouter from "./routes/admin";
import { logger } from "./lib/logger";

const app: Express = express();

app.use(
  pinoHttp({
    logger,
    serializers: {
      req(req) {
        return {
          id: req.id,
          method: req.method,
          url: req.url?.split("?")[0],
        };
      },
      res(res) {
        return {
          statusCode: res.statusCode,
        };
      },
    },
  }),
);
app.use(cors());
app.use(express.json({ limit: "20mb" }));
app.use(express.urlencoded({ extended: true }));

// Anthropic-compatible proxy endpoints
app.use(messagesRouter);
// Admin / dashboard API
app.use(adminRouter);
// Existing /api routes (healthz, etc.)
app.use("/api", router);

// Static dashboard
const here = path.dirname(fileURLToPath(import.meta.url));
app.use(express.static(path.resolve(here, "public")));
app.get("/", (_req, res) => {
  res.sendFile(path.resolve(here, "public/index.html"));
});

export default app;
