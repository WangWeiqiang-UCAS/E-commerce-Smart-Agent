import { NextRequest } from "next/server";
import { z } from "zod";
import { assertSameOrigin, sessionTokens, unauthorized } from "@/lib/server-auth";
import { traceHeader, traceId, traceResponseHeaders } from "@/lib/trace";

const identifier = z.coerce.number().int().positive();
const objectKey = z.string().regex(/^after-sales\/[1-9][0-9]*\/[0-9a-f-]{36}\.(jpg|png|pdf)$/);

export async function GET(request: NextRequest, { params }: { params: Promise<{ afterSaleId: string }> }) {
  try {
    assertSameOrigin(request);
    const { afterSaleId } = await params;
    const { commerce } = await sessionTokens();
    const id = traceId(request);
    const key = objectKey.parse(request.nextUrl.searchParams.get("objectKey"));
    const response = await fetch(`${process.env.COMMERCE_API_URL}/returnApply/evidence/${identifier.parse(afterSaleId)}/download?objectKey=${encodeURIComponent(key)}`, {
      headers: { Authorization: `Bearer ${commerce}`, [traceHeader]: id }, cache: "no-store"
    });
    return new Response(await response.text(), { status: response.status, headers: traceResponseHeaders(id) });
  } catch (error) {
    return error instanceof z.ZodError || error instanceof Error && error.message.includes("origin")
      ? Response.json({ error: "invalid_request" }, { status: 400 }) : unauthorized();
  }
}
