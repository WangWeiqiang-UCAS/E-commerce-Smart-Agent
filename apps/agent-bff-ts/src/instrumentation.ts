import { registerOTel } from "@vercel/otel";

/**
 * Next.js invokes this once for each server instance before route handlers run.
 * @vercel/otel honours the standard OTEL_EXPORTER_OTLP_* environment variables
 * and instruments server-side fetch calls made by the BFF.
 */
export function register() {
  if (process.env.NEXT_RUNTIME === "nodejs") {
    registerOTel({ serviceName: process.env.OTEL_SERVICE_NAME ?? "agent-bff" });
  }
}
