import { NextRequest } from "next/server";
import { z } from "zod";
import { assertSameOrigin, sessionTokens, unauthorized } from "@/lib/server-auth";
import { traceHeader, traceId, traceResponseHeaders } from "@/lib/trace";

const identifier = z.coerce.number().int().positive();
const body = z.object({ objectKeys: z.array(z.string().regex(/^after-sales\/[1-9][0-9]*\/[0-9a-f-]{36}\.(jpg|png|pdf)$/)).min(1).max(3) });

export async function POST(request: NextRequest, { params }: { params: Promise<{ afterSaleId: string }> }) {
  try {
    assertSameOrigin(request);
    const { afterSaleId } = await params;
    const { commerce } = await sessionTokens();
    const id = traceId(request);
    const response = await fetch(`${process.env.COMMERCE_API_URL}/returnApply/evidence/${identifier.parse(afterSaleId)}`, {
      method: "POST",
      headers: { Authorization: `Bearer ${commerce}`, "Content-Type": "application/json", [traceHeader]: id },
      body: JSON.stringify(body.parse(await request.json())),
      cache: "no-store"
    });
    return new Response(await response.text(), { status: response.status, headers: traceResponseHeaders(id) });
  } catch (error) {
    return error instanceof z.ZodError || error instanceof Error && error.message.includes("origin")
      ? Response.json({ error: "invalid_request" }, { status: 400 }) : unauthorized();
  }
}
