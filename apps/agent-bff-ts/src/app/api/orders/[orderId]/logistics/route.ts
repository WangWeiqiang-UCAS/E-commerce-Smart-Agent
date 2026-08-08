import { NextRequest } from "next/server";
import { z } from "zod";
import { assertSameOrigin, sessionTokens, unauthorized } from "@/lib/server-auth";

const identifier = z.coerce.number().int().positive();

export async function GET(request: NextRequest, { params }: { params: Promise<{ orderId: string }> }) {
  try {
    assertSameOrigin(request);
    const { orderId } = await params;
    const { commerce } = await sessionTokens();
    const response = await fetch(`${process.env.COMMERCE_API_URL}/agent/commerce/orders/${identifier.parse(orderId)}/logistics`, {
      headers: { Authorization: `Bearer ${commerce}` }, cache: "no-store"
    });
    return new Response(await response.text(), { status: response.status, headers: { "Content-Type": "application/json" } });
  } catch (error) {
    return error instanceof z.ZodError ? Response.json({ error: "invalid_request" }, { status: 400 }) : unauthorized();
  }
}
