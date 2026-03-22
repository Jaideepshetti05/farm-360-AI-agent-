"use client";
import React, { useState, useRef, useEffect } from "react";
import { Send, Paperclip } from "lucide-react";

export default function ChatInput({ setMessages }: { setMessages: any }) {
  const [query, setQuery] = useState("");
  const [loading, setLoading] = useState(false);
  const [file, setFile] = useState<File | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const bottomRef = useRef<HTMLDivElement>(null);
  // Accumulator ref to avoid React StrictMode double-invoke duplicating streamed lines
  const contentAccRef = useRef("");

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [loading]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!query.trim() && !file) return;

    const previewUrl = file ? URL.createObjectURL(file) : undefined;

    // Add User message
    setMessages((prev: any) => [...prev, { role: "user", content: query, imagePreview: previewUrl }]);

    // Reset inputs
    const sentQuery = query;
    setQuery("");
    const uploadRef = file;
    setFile(null);
    setLoading(true);

    // Add empty assistant placeholder for streaming/loading
    setMessages((prev: any) => [...prev, { role: "assistant", content: "" }]);

    try {
      const formData = new FormData();
      if (sentQuery.trim()) formData.append("query", sentQuery);
      formData.append("session_id", "default_react_session");
      if (uploadRef) formData.append("image", uploadRef);

      const baseUrl = process.env.NEXT_PUBLIC_API_BASE_URL || "http://127.0.0.1:8000";
      const apiKey = process.env.NEXT_PUBLIC_FARM360_API_KEY || "secure-secret-key-1234";

      // Image analysis → direct JSON endpoint; text → streaming SSE
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

      // ── Image path: backend returns JSON directly ──
      if (uploadRef) {
        const resp = await response.json();
        // resp may already be a structured object (no wrapping key)
        const structured = resp.response || resp;
        const content =
          typeof structured === "string"
            ? structured
            : JSON.stringify(structured);

        setMessages((prev: any) => {
          const updated = [...prev];
          updated[updated.length - 1] = { ...updated[updated.length - 1], content };
          return updated;
        });
        setLoading(false);
        return;
      }

      // ── SSE Streaming path ──
      // The backend streams a JSON string line by line (each "data:" line is one
      // line of the pretty-printed JSON). We accumulate them and, once [DONE]
      // arrives, the content is the full JSON string that ChatCanvas will parse.

      const reader = response.body.getReader();
      const decoder = new TextDecoder("utf-8");
      let buffer = "";
      contentAccRef.current = "";

      // Throttled UI flush via requestAnimationFrame
      let rafId: number | null = null;
      const flush = () => {
        const snapshot = contentAccRef.current;
        setMessages((prev: any) => {
          const updated = [...prev];
          updated[updated.length - 1] = { ...updated[updated.length - 1], content: snapshot };
          return updated;
        });
        rafId = null;
      };

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
            if (data === "[DONE]") { reader.cancel(); break outer; }
            if (data.startsWith("[ERROR]")) {
              contentAccRef.current = `⚠️ ${data}`;
              flush();
              break outer;
            }
            contentAccRef.current += data;
            if (!rafId) rafId = requestAnimationFrame(flush);
          }
        }
      }
      // Final flush for remaining content
      flush();

    } catch (e: any) {
      console.error(e);
      setMessages((prev: any) => {
        const updated = [...prev];
        updated[updated.length - 1] = {
          ...updated[updated.length - 1],
          content: `⚠️ System Error: ${e.message}`,
        };
        return updated;
      });
    } finally {
      setLoading(false);
    }
  };

  const suggestions = ["Suggest better crops", "My crop leaves are yellow", "Best irrigation method"];

  return (
    <div className="absolute bottom-0 w-full p-4 bg-gradient-to-t from-gray-900 via-gray-900/95 to-transparent">

      {/* File Preview */}
      {file && (
        <div className="max-w-3xl mx-auto mb-2 p-2 bg-gray-800 rounded-lg inline-flex items-center gap-2 border border-gray-700">
          <span className="text-xs text-gray-300">📎 {file.name} attached</span>
          <button onClick={() => setFile(null)} className="text-red-400 font-bold ml-2 hover:text-red-300">✕</button>
        </div>
      )}

      {/* Main Form */}
      <form
        onSubmit={handleSubmit}
        className="max-w-3xl mx-auto relative flex items-center bg-gray-800 rounded-2xl p-2 md:p-3 border border-gray-700 shadow-2xl focus-within:border-green-700/70 transition-colors"
      >
        <input
          type="file"
          accept="image/*"
          className="hidden"
          ref={fileInputRef}
          onChange={(e) => {
            if (e.target.files && e.target.files[0]) setFile(e.target.files[0]);
          }}
        />

        <button
          type="button"
          onClick={() => fileInputRef.current?.click()}
          className="p-2 text-gray-400 hover:text-green-400 rounded-full hover:bg-gray-700 transition-colors"
          title="Attach crop image"
        >
          <Paperclip size={20} />
        </button>

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
          placeholder={loading ? "Farm360 Expert is analyzing..." : "Ask about crops, diseases, irrigation…"}
          disabled={loading}
          className="flex-1 bg-transparent border-none focus:outline-none text-white px-3 py-1 resize-none max-h-40 overflow-y-auto placeholder-gray-500"
        />

        <button
          type="submit"
          disabled={loading || (!query.trim() && !file)}
          className="p-2 ml-1 bg-green-600 hover:bg-green-500 rounded-full text-white disabled:opacity-30 transition-all"
          title="Send"
        >
          <Send size={18} fill="currentColor" />
        </button>
      </form>

      <p className="text-center text-xs text-gray-600 mt-2 hidden md:block">
        Farm360 AI · Always verify crop and veterinary recommendations locally.
      </p>
      <div ref={bottomRef} />
    </div>
  );
}
