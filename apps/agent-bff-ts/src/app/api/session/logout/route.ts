import { cookies } from "next/headers";
import { assertSameOrigin, commerceCookie, sessionCookieOptions } from "@/lib/server-auth";
import { NextRequest } from "next/server";

export async function POST(request: NextRequest) {
  try {
    assertSameOrigin(request);
    const store = await cookies();
    store.set(commerceCookie, "", { ...sessionCookieOptions, maxAge: 0 });
    return Response.json({ authenticated: false });
  } catch { return Response.json({ error: "invalid_request" }, { status: 400 }); }
}
