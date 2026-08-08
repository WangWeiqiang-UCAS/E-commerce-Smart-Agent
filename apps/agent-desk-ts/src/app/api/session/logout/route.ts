import { NextRequest } from "next/server";
import { cookies } from "next/headers";
import { adminCookie, assertSameOrigin, sessionCookieOptions } from "@/lib/admin-auth";

export async function POST(request: NextRequest) {
  try {
    assertSameOrigin(request);
    const store = await cookies();
    store.set(adminCookie, "", { ...sessionCookieOptions, maxAge: 0 });
    return Response.json({ authenticated: false });
  } catch {
    return Response.json({ error: "invalid_request" }, { status: 400 });
  }
}
