import { NextRequest } from "next/server";
import { z } from "zod";
import { assertSameOrigin, sessionTokens, unauthorized } from "@/lib/server-auth";
import { traceHeader, traceId, traceResponseHeaders } from "@/lib/trace";

const threadId = z.string().min(1).max(128);

export async function GET(request: NextRequest, context: { params: Promise<{ threadId: string }> }) {
  try {
    assertSameOrigin(request);
    const id = traceId(request);
    const { threadId: rawThreadId } = await context.params;
    const { commerce } = await sessionTokens();
    const response = await fetch(`${process.env.AGENT_RUNTIME_URL}/api/v1/chat/history/${encodeURIComponent(threadId.parse(rawThreadId))}`, {
      headers: {
        Authorization: `Bearer ${commerce}`,
        "X-BFF-Secret": process.env.BFF_TO_AGENT_SHARED_SECRET ?? "",
        [traceHeader]: id,
      },
      cache: "no-store",
    });
    return new Response(await response.text(), { status: response.status, headers: traceResponseHeaders(id) });
  } catch (error) {
    return error instanceof z.ZodError || error instanceof Error && error.message.includes("trace")
      ? Response.json({ error: "invalid_request" }, { status: 400 }) : unauthorized();
  }
}
