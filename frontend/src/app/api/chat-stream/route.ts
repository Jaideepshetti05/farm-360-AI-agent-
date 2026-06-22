import { NextRequest } from "next/server";

/**
 * Server-side proxy for /chat_stream.
 * Keeps FARM360_API_KEY off the client bundle entirely.
 */
export async function POST(req: NextRequest) {
  const BACKEND = process.env.NEXT_PUBLIC_API_BASE_URL || "http://127.0.0.1:8000";
  const API_KEY = process.env.FARM360_API_KEY || process.env.NEXT_PUBLIC_FARM360_API_KEY || "";

  try {
    const formData = await req.formData();

    // Forward the form data to the backend
    const backendRes = await fetch(`${BACKEND}/chat_stream`, {
      method: "POST",
      headers: { "X-API-Key": API_KEY },
      body: formData,
    });

    if (!backendRes.ok) {
      const errText = await backendRes.text();
      return new Response(JSON.stringify({ detail: errText }), {
        status: backendRes.status,
        headers: { "Content-Type": "application/json" },
      });
    }

    // Stream the SSE response through to the client
    return new Response(backendRes.body, {
      status: 200,
      headers: {
        "Content-Type": "text/event-stream",
        "Cache-Control": "no-cache, no-transform",
        Connection: "keep-alive",
        "X-Accel-Buffering": "no",
      },
    });
  } catch (err: unknown) {
    const message = err instanceof Error ? err.message : "Proxy error";
    console.error("[chat-stream proxy]", message);
    return new Response(JSON.stringify({ detail: message }), {
      status: 502,
      headers: { "Content-Type": "application/json" },
    });
  }
}
