import { cookies } from "next/headers";
import { NextRequest } from "next/server";

const adminCookie = process.env.ADMIN_SESSION_COOKIE ?? "desk_admin_token";

export const sessionCookieOptions = {
  httpOnly: true,
  sameSite: "lax" as const,
  secure: process.env.SESSION_COOKIE_SECURE !== "false",
  path: "/"
};

export { adminCookie };

export async function adminToken() {
  const token = (await cookies()).get(adminCookie)?.value;
  if (!token) throw new Error("Unauthenticated session");
  return token;
}

export function assertSameOrigin(request: NextRequest) {
  const expected = process.env.PUBLIC_APP_ORIGIN;
  const origin = request.headers.get("origin");
  if (expected && origin && origin !== expected) throw new Error("Cross-origin request rejected");
}

export function unauthorized() {
  return Response.json({ error: "unauthenticated" }, { status: 401 });
}
