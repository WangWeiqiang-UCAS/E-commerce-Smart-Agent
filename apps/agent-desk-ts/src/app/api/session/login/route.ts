import { NextRequest } from "next/server";
import { cookies } from "next/headers";
import { z } from "zod";
import { adminCookie, assertSameOrigin, sessionCookieOptions } from "@/lib/admin-auth";

const input = z.object({ username: z.string().min(3).max(50), password: z.string().min(6).max(256) });

export async function POST(request: NextRequest) {
  try {
    assertSameOrigin(request);
    const credentials = input.parse(await request.json());
    const response = await fetch(`${process.env.AGENT_RUNTIME_URL}/api/v1/login`, {
      method: "POST",
      headers: { "Content-Type": "application/json", "X-BFF-Secret": process.env.BFF_TO_AGENT_SHARED_SECRET ?? "" },
      body: JSON.stringify(credentials),
      cache: "no-store"
    });
    if (!response.ok) return Response.json({ error: "invalid_credentials" }, { status: 401 });
    const body = await response.json();
    if (!body?.is_admin || !body?.access_token) return Response.json({ error: "admin_required" }, { status: 403 });
    const store = await cookies();
    store.set(adminCookie, body.access_token, sessionCookieOptions);
    return Response.json({ authenticated: true, username: body.username });
  } catch (error) {
    return error instanceof z.ZodError ? Response.json({ error: "invalid_request" }, { status: 400 }) : Response.json({ error: "authentication_unavailable" }, { status: 503 });
  }
}
