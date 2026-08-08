import { NextResponse } from "next/server";

// Refund decisions belong to the Java after-sales state machine. The legacy
// Python approval queue has been retired and must not be recreated here.
export async function GET() {
  return NextResponse.json({
    tasks: [],
    notice: "The legacy approval queue is retired. Use the commerce after-sales console until its audited API is integrated here."
  });
}
