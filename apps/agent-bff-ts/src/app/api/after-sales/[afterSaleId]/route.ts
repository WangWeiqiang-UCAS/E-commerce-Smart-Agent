import { NextRequest } from "next/server";
import { z } from "zod";
import { assertSameOrigin, sessionTokens, unauthorized } from "@/lib/server-auth";

const identifier = z.coerce.number().int().positive();

export async function GET(request: NextRequest, { params }: { params: Promise<{ afterSaleId: string }> }) {
  try {
    assertSameOrigin(request);
    const { afterSaleId } = await params;
    const { commerce } = await sessionTokens();
    const response = await fetch(`${process.env.COMMERCE_API_URL}/agent/commerce/after-sales/${identifier.parse(afterSaleId)}`, {
      headers: { Authorization: `Bearer ${commerce}` }, cache: "no-store"
    });
    return new Response(await response.text(), { status: response.status, headers: { "Content-Type": "application/json" } });
  } catch (error) {
    return error instanceof z.ZodError ? Response.json({ error: "invalid_request" }, { status: 400 }) : unauthorized();
  }
}
