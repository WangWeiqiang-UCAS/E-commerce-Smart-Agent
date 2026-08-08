import { NextRequest } from "next/server";
import { assertSameOrigin, sessionTokens, unauthorized } from "@/lib/server-auth";
import { traceHeader, traceId, traceResponseHeaders } from "@/lib/trace";

export async function GET(request: NextRequest, context: { params: Promise<{ threadId: string }> }) {
  try {
    assertSameOrigin(request);
    const { threadId } = await context.params;
    const { agent } = await sessionTokens();
    const id = traceId(request);
    const response = await fetch(`${process.env.AGENT_RUNTIME_URL}/api/v1/chat/history/${encodeURIComponent(threadId)}`, {
      headers: { Authorization: `Bearer ${agent}`, "X-BFF-Secret": process.env.BFF_TO_AGENT_SHARED_SECRET ?? "", [traceHeader]: id },
      cache: "no-store"
    });
    if (!response.ok) return unauthorized();
    return new Response(await response.text(), { status: response.status, headers: traceResponseHeaders(id) });
  } catch {
    return unauthorized();
  }
}
