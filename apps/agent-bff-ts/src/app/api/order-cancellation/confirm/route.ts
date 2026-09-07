import { NextRequest } from "next/server";
import { z } from "zod";
import { assertSameOrigin, sessionTokens, unauthorized } from "@/lib/server-auth";
import { traceHeader, traceId, traceResponseHeaders } from "@/lib/trace";

const input = z.object({ actionToken: z.string().min(32), idempotencyKey: z.string().uuid(), previewHash: z.string().length(64), agentTraceId: z.string().uuid(), confirmedAt: z.string().datetime() });

export async function POST(request: NextRequest) {
  try {
    assertSameOrigin(request);
    const { commerce } = await sessionTokens();
    const body = input.parse(await request.json());
    const id = traceId(request, body.agentTraceId);
    const response = await fetch(`${process.env.COMMERCE_API_URL}/agent/commerce/actions/order-cancellation/confirm`, {
      method: "POST", headers: { Authorization: `Bearer ${commerce}`, "Content-Type": "application/json", [traceHeader]: id },
      body: JSON.stringify(body), cache: "no-store"
    });
    return new Response(await response.text(), { status: response.status, headers: traceResponseHeaders(id) });
  } catch (error) {
    return error instanceof z.ZodError || error instanceof Error && error.message.includes("trace")
      ? Response.json({ error: "invalid_request" }, { status: 400 }) : unauthorized();
  }
}
