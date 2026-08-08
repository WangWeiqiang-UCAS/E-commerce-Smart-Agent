import { NextRequest } from "next/server";
import { cookies } from "next/headers";
import { z } from "zod";
import { agentCookie, assertSameOrigin, commerceCookie, sessionCookieOptions } from "@/lib/server-auth";

const input = z.object({ username: z.string().min(3).max(50), password: z.string().min(6).max(256) });

export async function POST(request: NextRequest) {
  try {
    assertSameOrigin(request);
    const credentials = input.parse(await request.json());
    const commerceBody = new URLSearchParams(credentials).toString();
    const [agentResponse, commerceResponse] = await Promise.all([
      fetch(`${process.env.AGENT_RUNTIME_URL}/api/v1/login`, { method: "POST", headers: { "Content-Type": "application/json", "X-BFF-Secret": process.env.BFF_TO_AGENT_SHARED_SECRET ?? "" }, body: JSON.stringify(credentials), cache: "no-store" }),
      fetch(`${process.env.COMMERCE_API_URL}/sso/login`, { method: "POST", headers: { "Content-Type": "application/x-www-form-urlencoded" }, body: commerceBody, cache: "no-store" })
    ]);
    if (!agentResponse.ok || !commerceResponse.ok) return Response.json({ error: "invalid_credentials" }, { status: 401 });
    const agent = await agentResponse.json(); const commerce = await commerceResponse.json();
    const commerceToken = commerce?.data?.token;
    if (!agent?.access_token || !commerceToken || commerce?.code !== 200) return Response.json({ error: "invalid_credentials" }, { status: 401 });
    const store = await cookies();
    store.set(agentCookie, agent.access_token, sessionCookieOptions);
    store.set(commerceCookie, commerceToken, sessionCookieOptions);
    return Response.json({ authenticated: true, username: agent.username, is_admin: agent.is_admin });
  } catch (error) {
    return error instanceof z.ZodError ? Response.json({ error: "invalid_request" }, { status: 400 }) : Response.json({ error: "authentication_unavailable" }, { status: 503 });
  }
}
