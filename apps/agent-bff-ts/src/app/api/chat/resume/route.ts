import { NextRequest } from "next/server";
import { z } from "zod";
import { assertSameOrigin, sessionTokens, unauthorized } from "@/lib/server-auth";
import { traceHeader, traceId, traceResponseHeaders } from "@/lib/trace";

const input = z.object({
  thread_id: z.string().min(1).max(128),
  decision: z.enum(["confirmed", "cancelled"]),
  confirmed_at: z.string().datetime().optional(),
  approver_id: z.number().int().positive().optional(),
  idempotency_key: z.string().uuid().optional()
});

export async function POST(request: NextRequest) {
  try {
    assertSameOrigin(request);
    const { agent, commerce } = await sessionTokens();
    const body = input.parse(await request.json());
    const id = traceId(request);
    const response = await fetch(`${process.env.AGENT_RUNTIME_URL}/api/v1/chat/resume`, {
      method: "POST",
      headers: {
        Authorization: `Bearer ${agent}`,
        "X-Commerce-Access-Token": commerce,
        "X-BFF-Secret": process.env.BFF_TO_AGENT_SHARED_SECRET ?? "",
        [traceHeader]: id,
        "Content-Type": "application/json"
      },
      body: JSON.stringify(body),
      cache: "no-store"
    });
    if (!response.ok) return Response.json({ error: "resume_failed" }, { status: response.status || 502 });
    return new Response(await response.text(), { status: response.status, headers: traceResponseHeaders(id) });
  } catch (error) {
    return error instanceof z.ZodError ? Response.json({ error: "invalid_request" }, { status: 400 }) : unauthorized();
  }
}
