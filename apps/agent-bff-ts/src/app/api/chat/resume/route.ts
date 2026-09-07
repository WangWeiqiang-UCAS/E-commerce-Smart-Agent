import { NextRequest } from "next/server";
import { z } from "zod";
import { assertSameOrigin, sessionTokens, unauthorized } from "@/lib/server-auth";
import { traceHeader, traceId, traceResponseHeaders } from "@/lib/trace";

const input = z.object({
  thread_id: z.string().min(1).max(128),
  decision: z.enum(["confirmed", "cancelled"]),
  confirmed_at: z.string().datetime().optional(),
  idempotency_key: z.string().uuid().optional(),
});

export async function POST(request: NextRequest) {
  try {
    assertSameOrigin(request);
    const body = input.parse(await request.json());
    const id = traceId(request);
    const { commerce } = await sessionTokens();
    const response = await fetch(`${process.env.AGENT_RUNTIME_URL}/api/v1/chat/resume`, {
      method: "POST",
      headers: {
        Authorization: `Bearer ${commerce}`,
        "X-Commerce-Access-Token": commerce,
        "X-BFF-Secret": process.env.BFF_TO_AGENT_SHARED_SECRET ?? "",
        "Content-Type": "application/json",
        [traceHeader]: id,
      },
      body: JSON.stringify(body),
      cache: "no-store",
    });
    return new Response(await response.text(), { status: response.status, headers: traceResponseHeaders(id) });
  } catch (error) {
    return error instanceof z.ZodError || error instanceof Error && error.message.includes("trace")
      ? Response.json({ error: "invalid_request" }, { status: 400 }) : unauthorized();
  }
}
