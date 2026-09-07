import { NextRequest } from "next/server";
import { z } from "zod";
import { assertSameOrigin, sessionTokens, unauthorized } from "@/lib/server-auth";
import { traceHeader, traceId, traceResponseHeaders } from "@/lib/trace";

const productId = z.coerce.number().int().positive();

export async function GET(request: NextRequest, context: { params: Promise<{ productId: string }> }) {
  try {
    assertSameOrigin(request);
    const id = traceId(request);
    const { productId: rawProductId } = await context.params;
    const { commerce } = await sessionTokens();
    const response = await fetch(`${process.env.COMMERCE_API_URL}/agent/commerce/products/${productId.parse(rawProductId)}`, {
      headers: { Authorization: `Bearer ${commerce}`, [traceHeader]: id }, cache: "no-store",
    });
    return new Response(await response.text(), { status: response.status, headers: traceResponseHeaders(id) });
  } catch (error) {
    return error instanceof z.ZodError || error instanceof Error && error.message.includes("trace")
      ? Response.json({ error: "invalid_request" }, { status: 400 }) : unauthorized();
  }
}
