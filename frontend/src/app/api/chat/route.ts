import { NextRequest, NextResponse } from "next/server";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

export async function POST(req: NextRequest) {
  try {
    const formData = await req.formData();
    
    const backendUrl =
      process.env.BACKEND_API_URL ||
      process.env.NEXT_PUBLIC_API_BASE_URL ||
      "http://127.0.0.1:8000";
      
    const apiKey = process.env.FARM360_API_KEY || "";

    const backendResponse = await fetch(`${backendUrl}/chat_stream`, {
      method: "POST",
      headers: {
        "X-API-Key": apiKey,
      },
      body: formData,
    });

    if (!backendResponse.ok) {
      const errorText = await backendResponse.text();
      return NextResponse.json(
        { detail: errorText || "Backend streaming failed" },
        { status: backendResponse.status }
      );
    }

    if (!backendResponse.body) {
      return NextResponse.json(
        { detail: "No response body from backend stream" },
        { status: 500 }
      );
    }

    // Proxy the line-delimited SSE stream directly to the client
    return new NextResponse(backendResponse.body, {
      headers: {
        "Content-Type": "text/event-stream",
        "Cache-Control": "no-cache, no-transform",
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no",
      },
    });
  } catch (error: any) {
    console.error("[API Proxy Error]:", error);
    return NextResponse.json(
      { detail: error?.message || "Internal Proxy Error" },
      { status: 500 }
    );
  }
}
