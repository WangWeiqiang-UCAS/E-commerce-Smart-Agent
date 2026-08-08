import { NextRequest } from "next/server";
import { adminToken, assertSameOrigin, unauthorized } from "@/lib/admin-auth";

export async function GET(request: NextRequest) {
  try {
    assertSameOrigin(request);
    const token = await adminToken();
    const response = await fetch(`${process.env.AGENT_RUNTIME_URL}/api/v1/threads`, {
      headers: { Authorization: `Bearer ${token}`, "X-BFF-Secret": process.env.BFF_TO_AGENT_SHARED_SECRET ?? "" },
      cache: "no-store"
    });
    if (!response.ok) return unauthorized();
    return Response.json(await response.json());
  } catch {
    return unauthorized();
  }
}
