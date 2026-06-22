"use client";
import React, {
  useState, useRef, useEffect,
  useCallback, forwardRef, useImperativeHandle,
} from "react";
import { Send, Paperclip, X, Loader2, Wifi, WifiOff } from "lucide-react";
import type { Message } from "@/app/page";

// ─── Constants ──────────────────────────────────────────────────────────────
const MAX_RETRIES = 2;
const RETRY_DELAY_MS = 1500;
const REQUEST_TIMEOUT_MS = 300000; // 5 minutes for long streams

// ─── Public handle (used by parent to trigger suggestion sends) ──────────────
export type ChatInputHandle = { sendQuery: (q: string) => void };

// ─── Helper: delay ──────────────────────────────────────────────────────────
const delay = (ms: number) => new Promise(resolve => setTimeout(resolve, ms));

// ─── Helper: fetch with timeout ─────────────────────────────────────────────
async function fetchWithTimeout(
  url: string,
  options: RequestInit,
  timeoutMs: number = REQUEST_TIMEOUT_MS
): Promise<Response> {
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), timeoutMs);

  try {
    const response = await fetch(url, {
      ...options,
      signal: controller.signal,
    });
    clearTimeout(timeoutId);
    return response;
  } catch (error) {
    clearTimeout(timeoutId);
    throw error;
  }
}

// ─── Generate a session ID (SSR-safe: only call on client) ──────────────────
function getOrCreateSessionId(): string {
  if (typeof window === "undefined") {
    // Running on the server — return a temporary placeholder (will be replaced in useEffect)
    return `session_${Date.now()}_${Math.random().toString(36).slice(2, 9)}`;
  }
  const KEY = "farm360_session_id";
  let sid = localStorage.getItem(KEY);
  if (!sid) {
    sid = `session_${Date.now()}_${Math.random().toString(36).slice(2, 9)}`;
    localStorage.setItem(KEY, sid);
  }
  return sid;
}

// ─── Component ───────────────────────────────────────────────────────────────
const ChatInput = forwardRef<ChatInputHandle, {
  messages: Message[];
  setMessages: React.Dispatch<React.SetStateAction<Message[]>>;
  selectedModel: string;
}>(function ChatInput({ messages, setMessages, selectedModel }, ref) {
  const [query, setQuery]     = useState("");
  const [loading, setLoading] = useState(false);
  const [file, setFile]       = useState<File | null>(null);
  const [connectionStatus, setConnectionStatus] = useState<"online" | "offline" | "checking">("online");

  const fileInputRef   = useRef<HTMLInputElement>(null);
  const textareaRef    = useRef<HTMLTextAreaElement>(null);
  const contentAccRef  = useRef("");
  const currentIdRef   = useRef("");
  const loadingRef     = useRef(false); // sync ref for callback closures
  const retryCountRef  = useRef(0);
  const sessionIdRef   = useRef(""); // Populated in useEffect (client-only: localStorage)

  // ── Initialize sessionId from localStorage (client-only) ─────────────────
  useEffect(() => {
    sessionIdRef.current = getOrCreateSessionId();
  }, []);

  // ── Auto-resize textarea ─────────────────────────────────────────────────
  useEffect(() => {
    const ta = textareaRef.current;
    if (!ta) return;
    ta.style.height = "auto";
    ta.style.height = Math.min(ta.scrollHeight, 168) + "px";
  }, [query]);

  // ── Cleanup blob URLs when messages change ────────────────────────────────
  const prevBlobUrlsRef = useRef<Set<string>>(new Set());
  useEffect(() => {
    const currentBlobUrls = new Set<string>();
    messages.forEach(m => {
      if (m.imagePreview?.startsWith("blob:")) {
        currentBlobUrls.add(m.imagePreview);
      }
    });
    // Revoke URLs that are no longer in the message list
    prevBlobUrlsRef.current.forEach(url => {
      if (!currentBlobUrls.has(url)) {
        URL.revokeObjectURL(url);
      }
    });
    prevBlobUrlsRef.current = currentBlobUrls;
  }, [messages]);

  // ── Patch the last assistant message by id ───────────────────────────────
  const patchMsg = useCallback((patch: Partial<Message>) => {
    const id = currentIdRef.current;
    setMessages(prev => {
      const idx = prev.findIndex(m => m.id === id);
      if (idx === -1) return prev;
      const next = [...prev];
      next[idx] = { ...next[idx], ...patch };
      return next;
    });
  }, [setMessages]);

  // ── Core submit logic with retry ────────────────────────────────────────────
  const doSubmit = useCallback(async (submittedQuery: string, submittedFile: File | null) => {
    if (loadingRef.current) return;
    if (!submittedQuery.trim() && !submittedFile) return;

    const uid   = `u-${Date.now()}`;
    const aid   = `a-${Date.now() + 1}`;
    currentIdRef.current  = aid;
    contentAccRef.current = "";
    loadingRef.current    = true;
    retryCountRef.current = 0;

    const preview = submittedFile ? URL.createObjectURL(submittedFile) : undefined;

    // Add user + placeholder assistant messages
    setMessages(prev => [
      ...prev,
      { id: uid, role: "user", content: submittedQuery, imagePreview: preview, streaming: false, timestamp: new Date() },
      { id: aid, role: "assistant", content: "", streaming: true, timestamp: new Date() },
    ]);

    setLoading(true);
    setQuery("");
    setFile(null);
    setConnectionStatus("checking");

    // All API calls go through Next.js server-side proxy routes.
    // The proxy injects the API key server-side — never exposed to client.

    // Retry loop
    while (retryCountRef.current <= MAX_RETRIES) {
      try {
        const form = new FormData();
        form.append("query", submittedQuery || "Analyze this image");
        form.append("session_id", sessionIdRef.current);
        form.append("model", selectedModel);
        if (submittedFile) form.append("image", submittedFile);

        const endpoint = submittedFile
          ? "/api/analyze-image"
          : "/api/chat-stream";

        const res = await fetchWithTimeout(endpoint, {
          method: "POST",
          body: form,
        }, REQUEST_TIMEOUT_MS);

        if (!res.ok) {
          const err = await res.text();
          throw new Error(`Server ${res.status}: ${err}`);
        }
        if (!res.body) throw new Error("No stream from server.");

        setConnectionStatus("online");

        // ── Image endpoint: JSON response ──────────────────────────────────
        if (submittedFile) {
          const json = await res.json();
          const data = json.response ?? json;
          let text: string;
          if (typeof data === "string") {
            text = data;
          } else {
            // Convert structured JSON to readable markdown
            const parts: string[] = [];
            if (data.summary) parts.push(`**${data.summary}**\n`);
            if (data.analysis) parts.push(data.analysis + "\n");
            if (data.recommendations?.length) {
              parts.push("## Recommendations");
              data.recommendations.forEach((r: string) => parts.push(`- ${r}`));
              parts.push("");
            }
            if (data.action_steps?.length) {
              parts.push("## Action Steps");
              data.action_steps.forEach((s: string, i: number) => parts.push(`${i + 1}. ${s}`));
            }
            if (data.missing_data_warning) parts.push(`\n> ⚠️ ${data.missing_data_warning}`);
            text = parts.join("\n");
          }
          patchMsg({ content: text, streaming: false });
          setLoading(false);
          loadingRef.current = false;
          return;
        }

        // ── SSE text/event-stream: update state on EVERY token ────────────
        const reader  = res.body.getReader();
        const decoder = new TextDecoder("utf-8");
        let buf = "";
        let streamError = false;

        try {
          outer: while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            buf += decoder.decode(value, { stream: true });

            let boundary: number;
            while ((boundary = buf.indexOf("\n\n")) !== -1) {
              const raw = buf.slice(0, boundary).trim();
              buf = buf.slice(boundary + 2);

              if (!raw.startsWith("data: ")) continue;
              const data = raw.slice(6).replace(/\\n/g, "\n");

              if (data === "[DONE]") {
                break outer;
              }

              if (data.startsWith("[ERROR]")) {
                streamError = true;
                patchMsg({
                  content: contentAccRef.current + `\n\n⚠️ ${data.slice(7).trim()}`,
                  streaming: false,
                });
                break outer;
              }

              // ✅ Real-time token append
              contentAccRef.current += data;
              patchMsg({ content: contentAccRef.current, streaming: true });
            }
          }
        } finally {
          // Ensure reader is always closed
          try { await reader.cancel(); } catch {}
        }

        // Ensure final state is non-streaming
        if (!streamError) {
          patchMsg({ streaming: false });
        }
        setLoading(false);
        loadingRef.current = false;
        return; // Success - exit the function

      } catch (err: unknown) {
        const error = err as Error;
        console.error(`[Farm360] Attempt ${retryCountRef.current + 1} failed:`, error.message);

        // Check if it's an abort (timeout) or network error
        const isTimeout = error.name === "AbortError";
        const isNetworkError = error.message?.includes("fetch") || error.message?.includes("network");

        if ((isTimeout || isNetworkError) && retryCountRef.current < MAX_RETRIES) {
          retryCountRef.current++;
          setConnectionStatus("checking");
          patchMsg({
            content: `⏳ Connection issue. Retrying... (${retryCountRef.current}/${MAX_RETRIES})`,
            streaming: true,
          });
          await delay(RETRY_DELAY_MS * retryCountRef.current); // Exponential backoff
          continue; // Retry
        }

        // Final failure
        setConnectionStatus("offline");
        const errorType = isTimeout ? "Request timed out" : isNetworkError ? "Network error" : "Error";
        patchMsg({
          content: `⚠️ **${errorType}**\n\n${error.message}\n\n**Troubleshooting:**\n- Check if backend is running on port 8000\n- Verify your internet connection\n- Try again in a few seconds`,
          streaming: false,
        });
        setLoading(false);
        loadingRef.current = false;
        return;
      }
    }
  }, [patchMsg, selectedModel, setMessages]);

  // ── Form submit ──────────────────────────────────────────────────────────
  const handleSubmit = (e?: React.FormEvent) => {
    e?.preventDefault();
    doSubmit(query, file);
  };

  // ── Expose sendQuery to parent (for suggestion chips) ────────────────────
  useImperativeHandle(ref, () => ({
    sendQuery: (q: string) => doSubmit(q, null),
  }));

  return (
    <div
      className="absolute bottom-0 inset-x-0 px-4 pb-4 pt-2"
      style={{ background: "linear-gradient(to top, var(--bg) 70%, transparent)" }}
    >
      {/* File preview badge */}
      {file && (
        <div className="max-w-3xl mx-auto mb-2">
          <span
            className="inline-flex items-center gap-2 px-3 py-1.5 rounded-lg text-sm"
            style={{ background: "var(--surface-2)", border: "1px solid var(--border-2)", color: "#aaa" }}
          >
            <span>📎</span>
            <span className="truncate max-w-[180px]">{file.name}</span>
            <button onClick={() => setFile(null)} className="ml-1 hover:text-red-400 transition-colors">
              <X size={13} />
            </button>
          </span>
        </div>
      )}

      {/* Input form */}
      <form
        onSubmit={handleSubmit}
        className="chat-form max-w-3xl mx-auto flex items-end gap-2 px-3 py-2.5 rounded-2xl"
        style={{ background: "var(--surface-2)", border: "1px solid var(--border-2)" }}
      >
        {/* Hidden file input */}
        <input
          type="file"
          accept="image/*"
          className="hidden"
          ref={fileInputRef}
          onChange={e => { if (e.target.files?.[0]) setFile(e.target.files[0]); }}
        />

        {/* Attach */}
        <button
          type="button"
          onClick={() => fileInputRef.current?.click()}
          title="Attach crop image for disease diagnosis"
          disabled={loading}
          className="shrink-0 mb-0.5 p-1.5 rounded-lg transition-all hover:bg-white/8 disabled:opacity-30"
          style={{ color: "#666" }}
          onMouseEnter={e => (e.currentTarget.style.color = "#4ade80")}
          onMouseLeave={e => (e.currentTarget.style.color = "#666")}
        >
          <Paperclip size={17} />
        </button>

        {/* Textarea */}
        <textarea
          ref={textareaRef}
          rows={1}
          value={query}
          onChange={e => setQuery(e.target.value)}
          onKeyDown={e => {
            if (e.key === "Enter" && !e.shiftKey) {
              e.preventDefault();
              handleSubmit();
            }
          }}
          placeholder={loading ? "Farm360 AI is thinking…" : "Ask about crops, diseases, soil, irrigation, livestock…"}
          disabled={loading}
          className="flex-1 bg-transparent border-none outline-none text-sm px-2 py-1 resize-none placeholder-[#444] leading-relaxed"
          style={{ color: "#e8e8e8", minHeight: "24px", maxHeight: "168px" }}
        />

        {/* Send */}
        <button
          type="submit"
          disabled={loading || (!query.trim() && !file)}
          title="Send (Enter)"
          className="shrink-0 mb-0.5 p-2.5 rounded-xl text-white transition-all disabled:opacity-25 disabled:cursor-not-allowed flex items-center justify-center"
          style={{ background: loading || (!query.trim() && !file) ? "#1e3a1e" : "var(--accent)" }}
        >
          {loading
            ? <Loader2 size={16} className="animate-spin" />
            : <Send size={15} />
          }
        </button>
      </form>

      <p className="text-center mt-2 text-[10px] hidden md:flex items-center justify-center gap-2" style={{ color: "#333" }}>
        <span className="flex items-center gap-1">
          {connectionStatus === "online" && <Wifi size={10} style={{ color: "#4ade80" }} />}
          {connectionStatus === "offline" && <WifiOff size={10} style={{ color: "#ef4444" }} />}
          {connectionStatus === "checking" && <Loader2 size={10} className="animate-spin" style={{ color: "#facc15" }} />}
          <span style={{ color: connectionStatus === "online" ? "#4ade80" : connectionStatus === "offline" ? "#ef4444" : "#facc15" }}>
            {connectionStatus === "online" ? "Connected" : connectionStatus === "offline" ? "Disconnected" : "Connecting..."}
          </span>
        </span>
        <span>·</span>
        <span>Farm360 AI · Verify critical advice with an agronomist</span>
      </p>
    </div>
  );
});

export default ChatInput;