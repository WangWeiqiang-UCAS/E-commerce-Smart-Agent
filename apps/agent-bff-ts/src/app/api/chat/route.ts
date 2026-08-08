import { NextRequest } from "next/server";
import { z } from "zod";
import { assertSameOrigin, sessionTokens, unauthorized } from "@/lib/server-auth";
import { traceHeader, traceId, traceResponseHeaders } from "@/lib/trace";

const input = z.object({ question: z.string().min(1).max(4000), thread_id: z.string().min(1).max(128) });

export async function POST(request: NextRequest) {
  try {
    assertSameOrigin(request);
    const id = traceId(request);
    const body = input.parse(await request.json());
    const { agent, commerce } = await sessionTokens();
    const response = await fetch(`${process.env.AGENT_RUNTIME_URL}/api/v1/chat`, {
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
    if (!response.ok || !response.body) return Response.json({ error: "agent_unavailable" }, { status: response.status || 502 });
    return new Response(response.body, { headers: { ...traceResponseHeaders(id, "text/event-stream"), "Cache-Control": "no-cache" } });
  } catch (error) {
    return error instanceof z.ZodError || error instanceof Error && error.message.includes("trace") ? Response.json({ error: "invalid_request" }, { status: 400 }) : unauthorized();
  }
}
