import { NextRequest } from "next/server";
import { z } from "zod";
import { assertSameOrigin, sessionTokens, unauthorized } from "@/lib/server-auth";
import { traceHeader, traceId, traceResponseHeaders } from "@/lib/trace";

const input = z.object({
  query: z.string().trim().min(1).max(200),
  pageNum: z.coerce.number().int().min(1).max(100).default(1),
  pageSize: z.coerce.number().int().min(1).max(20).default(10),
  sort: z.coerce.number().int().min(0).max(4).default(0),
});

export async function GET(request: NextRequest) {
  try {
    assertSameOrigin(request);
    const id = traceId(request);
    const parsed = input.parse(Object.fromEntries(request.nextUrl.searchParams));
    const { commerce } = await sessionTokens();
    const params = new URLSearchParams(Object.entries(parsed).map(([key, value]) => [key, String(value)]));
    const response = await fetch(`${process.env.COMMERCE_API_URL}/agent/commerce/products/search?${params}`, {
      headers: { Authorization: `Bearer ${commerce}`, [traceHeader]: id }, cache: "no-store",
    });
    return new Response(await response.text(), { status: response.status, headers: traceResponseHeaders(id) });
  } catch (error) {
    return error instanceof z.ZodError || error instanceof Error && error.message.includes("trace")
      ? Response.json({ error: "invalid_request" }, { status: 400 }) : unauthorized();
  }
}
