export async function GET() {
  const checks = await Promise.allSettled([
    fetch(`${process.env.AGENT_RUNTIME_URL}/health`, { cache: "no-store", signal: AbortSignal.timeout(3000) }),
    fetch(`${process.env.COMMERCE_API_URL}/actuator/health`, { cache: "no-store", signal: AbortSignal.timeout(3000) })
  ]);
  const agent = checks[0].status === "fulfilled" && checks[0].value.ok;
  const commerce = checks[1].status === "fulfilled" && checks[1].value.ok;
  const status = agent && commerce ? 200 : 503;
  return Response.json({ status: status === 200 ? "healthy" : "degraded", dependencies: { agent, commerce } }, { status });
}
