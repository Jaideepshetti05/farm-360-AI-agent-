"use client";
import React, { useState, useRef, useEffect } from "react";
import { Send, Paperclip } from "lucide-react";

type Message = {
  role: "user" | "assistant";
  content: string;
  imagePreview?: string;
  streaming?: boolean;
};

export default function ChatInput({
  setMessages,
}: {
  setMessages: React.Dispatch<React.SetStateAction<Message[]>>;
}) {
  const [query, setQuery] = useState("");
  const [loading, setLoading] = useState(false);
  const [file, setFile] = useState<File | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const bottomRef = useRef<HTMLDivElement>(null);

  // We accumulate the full JSON string in a ref — never set partial state
  const contentAccRef = useRef("");

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [loading]);

  // Helper: update the last assistant message
  const updateLastMsg = (patch: Partial<Message>) => {
    setMessages((prev) => {
      const updated = [...prev];
      updated[updated.length - 1] = { ...updated[updated.length - 1], ...patch };
      return updated;
    });
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!query.trim() && !file) return;

    const sentQuery = query;
    const previewUrl = file ? URL.createObjectURL(file) : undefined;
    const uploadRef = file;

    // Append user message
    setMessages((prev) => [
      ...prev,
      { role: "user", content: sentQuery, imagePreview: previewUrl },
    ]);

    // Reset inputs
    setQuery("");
    setFile(null);
    setLoading(true);

    // Append streaming placeholder — streaming:true hides partial JSON
    setMessages((prev) => [
      ...prev,
      { role: "assistant", content: "", streaming: true },
    ]);

    try {
      const formData = new FormData();
      if (sentQuery.trim()) formData.append("query", sentQuery);
      formData.append("session_id", "default_react_session");
      if (uploadRef) formData.append("image", uploadRef);

      const baseUrl =
        process.env.NEXT_PUBLIC_API_BASE_URL || "http://127.0.0.1:8000";
      const apiKey =
        process.env.NEXT_PUBLIC_FARM360_API_KEY || "secure-secret-key-1234";

      const endpoint = uploadRef
        ? `${baseUrl}/analyze_image`
        : `${baseUrl}/chat_stream`;

      const response = await fetch(endpoint, {
        method: "POST",
        headers: { "X-API-Key": apiKey },
        body: formData,
      });

      if (!response.ok) {
        const errText = await response.text();
        throw new Error(`HTTP ${response.status}: ${errText}`);
      }

      if (!response.body) throw new Error("No readable stream available.");

      // ── Image path: backend returns structured JSON directly ──────────────
      if (uploadRef) {
        const resp = await response.json();
        // resp is already the structured dict (or potentially { response: ... })
        const structured = resp.response ?? resp;
        const content =
          typeof structured === "string"
            ? structured
            : JSON.stringify(structured);
        // streaming: false so ChatCanvas will parse and render the structured card
        updateLastMsg({ content, streaming: false });
        setLoading(false);
        return;
      }

      // ── SSE Streaming path ────────────────────────────────────────────────
      // The backend streams the JSON pretty-printed line by line.
      // We accumulate ALL lines in a ref and only commit to React state when
      // [DONE] arrives — this prevents partial JSON from ever being rendered.
      const reader = response.body.getReader();
      const decoder = new TextDecoder("utf-8");
      let buffer = "";
      contentAccRef.current = "";

      outer: while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        let boundary: number;

        while ((boundary = buffer.indexOf("\n\n")) !== -1) {
          const message = buffer.slice(0, boundary).trim();
          buffer = buffer.slice(boundary + 2);

          if (message.startsWith("data: ")) {
            const data = message.slice(6).replace(/\\n/g, "\n");

            if (data === "[DONE]") {
              reader.cancel();
              // ✅ Commit the full accumulated JSON — now parseable
              updateLastMsg({
                content: contentAccRef.current,
                streaming: false,
              });
              break outer;
            }

            if (data.startsWith("[ERROR]")) {
              updateLastMsg({ content: `⚠️ ${data}`, streaming: false });
              break outer;
            }

            // Accumulate silently — DO NOT update React state here
            contentAccRef.current += data;
          }
        }
      }
    } catch (e: any) {
      console.error("Farm360 fetch error:", e);
      updateLastMsg({
        content: `⚠️ System Error: ${e.message}`,
        streaming: false,
      });
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="absolute bottom-0 w-full p-4 bg-gradient-to-t from-gray-900 via-gray-900/95 to-transparent">

      {/* File attachment preview */}
      {file && (
        <div className="max-w-3xl mx-auto mb-2 px-3 py-2 bg-gray-800 rounded-lg inline-flex items-center gap-2 border border-gray-700">
          <span className="text-xs text-gray-300">📎 {file.name}</span>
          <button
            onClick={() => setFile(null)}
            className="text-red-400 hover:text-red-300 font-bold ml-1"
          >
            ✕
          </button>
        </div>
      )}

      {/* Input form */}
      <form
        onSubmit={handleSubmit}
        className="max-w-3xl mx-auto flex items-center bg-gray-800 rounded-2xl px-3 py-2.5 border border-gray-700 shadow-2xl focus-within:border-green-700/70 transition-colors"
      >
        {/* Hidden file input */}
        <input
          type="file"
          accept="image/*"
          className="hidden"
          ref={fileInputRef}
          onChange={(e) => {
            if (e.target.files?.[0]) setFile(e.target.files[0]);
          }}
        />

        {/* Attach button */}
        <button
          type="button"
          onClick={() => fileInputRef.current?.click()}
          title="Attach crop image"
          className="p-1.5 text-gray-400 hover:text-green-400 rounded-full hover:bg-gray-700 transition-colors shrink-0"
        >
          <Paperclip size={19} />
        </button>

        {/* Text area */}
        <textarea
          rows={1}
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && !e.shiftKey) {
              e.preventDefault();
              handleSubmit(e);
            }
          }}
          placeholder={
            loading
              ? "Farm360 Expert is analyzing…"
              : "Ask about crops, diseases, irrigation…"
          }
          disabled={loading}
          className="flex-1 bg-transparent border-none focus:outline-none text-white text-sm px-3 py-1 resize-none max-h-40 overflow-y-auto placeholder-gray-500"
        />

        {/* Send button */}
        <button
          type="submit"
          disabled={loading || (!query.trim() && !file)}
          title="Send"
          className="p-2 ml-1 bg-green-600 hover:bg-green-500 disabled:opacity-30 rounded-full text-white transition-all shrink-0"
        >
          <Send size={17} fill="currentColor" />
        </button>
      </form>

      <p className="text-center text-[11px] text-gray-600 mt-2 hidden md:block">
        Farm360 AI · Verify crop and veterinary recommendations with local experts.
      </p>

      <div ref={bottomRef} />
    </div>
  );
}
