import { cookies } from "next/headers";
import { NextRequest } from "next/server";

const commerceCookie = process.env.COMMERCE_SESSION_COOKIE ?? "commerce_access_token";

export const sessionCookieOptions = {
  httpOnly: true,
  sameSite: "lax" as const,
  secure: process.env.SESSION_COOKIE_SECURE !== "false",
  path: "/"
};

export { commerceCookie };

export async function sessionTokens() {
  const store = await cookies();
  const commerce = store.get(commerceCookie)?.value;
  if (!commerce) throw new Error("Unauthenticated session");
  return { commerce };
}

export function assertSameOrigin(request: NextRequest) {
  const expected = process.env.PUBLIC_APP_ORIGIN;
  const origin = request.headers.get("origin");
  if (expected && origin && origin !== expected) throw new Error("Cross-origin request rejected");
}

export function unauthorized() {
  return Response.json({ error: "unauthenticated" }, { status: 401 });
}
