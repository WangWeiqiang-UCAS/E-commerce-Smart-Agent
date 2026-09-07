import { NextRequest } from "next/server";
import { assertSameOrigin, sessionTokens, unauthorized } from "@/lib/server-auth";
import { traceHeader, traceId, traceResponseHeaders } from "@/lib/trace";

const allowedTypes = new Set(["image/jpeg", "image/png", "application/pdf"]);
const maxFileSize = 5 * 1024 * 1024;

export async function POST(request: NextRequest) {
  try {
    assertSameOrigin(request);
    const form = await request.formData();
    const file = form.get("file");
    if (!(file instanceof File) || file.size === 0 || file.size > maxFileSize || !allowedTypes.has(file.type)) {
      return Response.json({ error: "invalid_file" }, { status: 400 });
    }
    const { commerce } = await sessionTokens();
    const id = traceId(request);
    const payload = new FormData();
    payload.append("file", file, file.name || "evidence");
    const response = await fetch(`${process.env.COMMERCE_API_URL}/returnApply/evidence/upload`, {
      method: "POST",
      headers: { Authorization: `Bearer ${commerce}`, [traceHeader]: id },
      body: payload,
      cache: "no-store"
    });
    return new Response(await response.text(), { status: response.status, headers: traceResponseHeaders(id) });
  } catch (error) {
    return error instanceof Error && error.message.includes("origin")
      ? Response.json({ error: "invalid_request" }, { status: 400 }) : unauthorized();
  }
}
