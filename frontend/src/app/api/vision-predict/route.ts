import { NextRequest } from "next/server";

/**
 * Server-side proxy for /vision/* endpoints.
 * Keeps FARM360_API_KEY off the client bundle entirely.
 */
export async function POST(req: NextRequest) {
  const BACKEND = process.env.NEXT_PUBLIC_API_BASE_URL || "http://127.0.0.1:8000";
  const API_KEY = process.env.FARM360_API_KEY || process.env.NEXT_PUBLIC_FARM360_API_KEY || "";

  try {
    const { searchParams } = new URL(req.url);
    const task = searchParams.get("task") || "crop-disease";

    // Read the form data from the incoming request
    const formData = await req.formData();

    const backendRes = await fetch(`${BACKEND}/vision/${task}`, {
      method: "POST",
      headers: { 
        "X-API-Key": API_KEY 
      },
      body: formData,
    });

    if (!backendRes.ok) {
      const errText = await backendRes.text();
      return new Response(JSON.stringify({ detail: errText }), {
        status: backendRes.status,
        headers: { "Content-Type": "application/json" },
      });
    }

    const data = await backendRes.json();
    return new Response(JSON.stringify(data), {
      status: 200,
      headers: { "Content-Type": "application/json" },
    });
  } catch (err: unknown) {
    const message = err instanceof Error ? err.message : "Proxy error";
    console.error("[vision-predict proxy]", message);
    return new Response(JSON.stringify({ detail: message }), {
      status: 502,
      headers: { "Content-Type": "application/json" },
    });
  }
}
