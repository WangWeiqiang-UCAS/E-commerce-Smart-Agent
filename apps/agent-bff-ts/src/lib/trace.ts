import { randomUUID } from "crypto";
import { NextRequest } from "next/server";

export const traceHeader = "X-Trace-Id";
const uuid = /^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i;

export function traceId(request: NextRequest, expected?: string) {
  const incoming = request.headers.get(traceHeader);
  if (incoming && !uuid.test(incoming)) throw new Error("Invalid trace ID");
  if (expected && incoming && incoming !== expected) throw new Error("Trace ID mismatch");
  return expected ?? incoming ?? randomUUID();
}

export function traceResponseHeaders(id: string, contentType = "application/json") {
  return { "Content-Type": contentType, [traceHeader]: id };
}
