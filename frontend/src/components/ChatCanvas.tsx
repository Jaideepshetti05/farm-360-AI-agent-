"use client";
import React, { useEffect, useRef, useCallback, useMemo } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { Copy, Check, Leaf, User } from "lucide-react";
import type { Message } from "@/app/page";

// ─── Copy button ────────────────────────────────────────────────────────────
function CopyButton({ text }: { text: string }) {
  const [copied, setCopied] = React.useState(false);
  const handleCopy = useCallback(async () => {
    await navigator.clipboard.writeText(text);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  }, [text]);
  return (
    <button
      onClick={handleCopy}
      title="Copy"
      className="copy-btn inline-flex items-center gap-1 px-2 py-1 rounded-md text-[11px] text-[var(--text-muted)] hover:text-white hover:bg-white/8 transition-all"
    >
      {copied ? (
        <><Check size={11} className="text-green-400" /><span className="text-green-400">Copied</span></>
      ) : (
        <><Copy size={11} /><span>Copy</span></>
      )}
    </button>
  );
}

// ─── Typing indicator ───────────────────────────────────────────────────────
const TypingDots = React.memo(function TypingDots() {
  return (
    <div className="flex items-center gap-1.5 py-1.5 px-0.5">
      <span className="w-2 h-2 rounded-full bg-green-500 dot-1" />
      <span className="w-2 h-2 rounded-full bg-green-500 dot-2" />
      <span className="w-2 h-2 rounded-full bg-green-500 dot-3" />
    </div>
  );
});

// ─── Markdown renderer (memoized for performance) ────────────────────────────
const Markdown = React.memo(function Markdown({ content, streaming }: { content: string; streaming?: boolean }) {
  return (
    <div className={`prose${streaming && content ? " streaming-cursor" : ""}`}>
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        components={{
          // Headings
          h1: ({ children }) => <h1 style={{ fontSize: "1.05rem", fontWeight: 700, color: "#f2f2f2", margin: "1em 0 0.4em" }}>{children}</h1>,
          h2: ({ children }) => <h2 style={{ fontSize: "0.95rem", fontWeight: 600, color: "#ebebeb", margin: "0.9em 0 0.35em" }}>{children}</h2>,
          h3: ({ children }) => <h3 style={{ fontSize: "0.88rem", fontWeight: 600, color: "#e0e0e0", margin: "0.8em 0 0.3em" }}>{children}</h3>,
          // Paragraphs
          p: ({ children }) => <p style={{ margin: "0.45em 0", color: "#d4d4d4", lineHeight: 1.65 }}>{children}</p>,
          // Unordered list
          ul: ({ children }) => <ul style={{ listStyle: "none", padding: 0, margin: "0.4em 0" }}>{children}</ul>,
          // Ordered list
          ol: ({ children }) => <ol style={{ listStyle: "none", padding: 0, margin: "0.4em 0", counterReset: "steps" }}>{children}</ol>,
          // List items — handle both ul and ol
          li: ({ children, ...props }) => {
            const listProps = props as React.LiHTMLAttributes<HTMLLIElement> & { ordered?: boolean; index?: number };
            const ordered = listProps.ordered;
            return ordered ? (
              <li style={{ display: "flex", gap: "0.55em", alignItems: "flex-start", margin: "0.25em 0", counterIncrement: "steps" }}>
                <span style={{ display: "inline-flex", alignItems: "center", justifyContent: "center", width: 18, height: 18, borderRadius: "50%", background: "rgba(22,163,74,0.18)", color: "#4ade80", fontSize: "0.68rem", fontWeight: 700, flexShrink: 0, marginTop: "0.12em" }}>
                  {(listProps.index || 0) + 1}
                </span>
                <span>{children}</span>
              </li>
            ) : (
              <li style={{ display: "flex", gap: "0.55em", alignItems: "flex-start", margin: "0.25em 0" }}>
                <span style={{ width: 5, height: 5, borderRadius: "50%", background: "#16a34a", marginTop: "0.48em", flexShrink: 0, display: "block" }} />
                <span>{children}</span>
              </li>
            );
          },
          // Strong/emphasis
          strong: ({ children }) => <strong style={{ fontWeight: 600, color: "#f0f0f0" }}>{children}</strong>,
          em: ({ children }) => <em style={{ color: "#999", fontStyle: "italic" }}>{children}</em>,
          // Code
          code: ({ className, children }) => {
            const isBlock = !!className;
            return isBlock ? (
              <code style={{ fontFamily: "'Courier New', Consolas, monospace", fontSize: "0.81em", color: "#86efac" }}>{children}</code>
            ) : (
              <code style={{ background: "rgba(255,255,255,0.07)", border: "1px solid #2e2e2e", borderRadius: 4, padding: "0.1em 0.38em", fontFamily: "'Courier New', Consolas, monospace", fontSize: "0.8em", color: "#86efac" }}>{children}</code>
            );
          },
          pre: ({ children }) => (
            <pre style={{ background: "#111", border: "1px solid #242424", borderRadius: 10, padding: "12px 14px", overflowX: "auto", margin: "0.7em 0" }}>{children}</pre>
          ),
          blockquote: ({ children }) => (
            <blockquote style={{ borderLeft: "2px solid #16a34a", margin: "0.5em 0", padding: "0.25em 0.75em", color: "#888" }}>{children}</blockquote>
          ),
          hr: () => <hr style={{ border: "none", borderTop: "1px solid #242424", margin: "0.9em 0" }} />,
          table: ({ children }) => (
            <div style={{ overflowX: "auto", margin: "0.7em 0" }}>
              <table style={{ width: "100%", borderCollapse: "collapse", fontSize: "0.81em" }}>{children}</table>
            </div>
          ),
          th: ({ children }) => <th style={{ background: "#1e1e1e", border: "1px solid #242424", padding: "5px 9px", textAlign: "left", fontWeight: 600, color: "#ccc" }}>{children}</th>,
          td: ({ children }) => <td style={{ border: "1px solid #242424", padding: "5px 9px", color: "#bbb" }}>{children}</td>,
        }}
      >
        {content}
      </ReactMarkdown>
    </div>
  );
}, (prevProps, nextProps) => {
  // Custom comparison - only re-render if content changed significantly
  // This prevents re-renders during streaming when only streaming prop changes
  if (prevProps.content !== nextProps.content) return false;
  // Don't re-render just because streaming changed from true to true
  if (prevProps.streaming === nextProps.streaming) return true;
  // Re-render when streaming ends (true -> false)
  if (prevProps.streaming && !nextProps.streaming) return false;
  return true;
});

// ─── Suggestion chips ───────────────────────────────────────────────────────
const SUGGESTIONS = [
  { icon: "🌾", short: "Best crops for June", full: "What are the best crops to plant in June in Punjab, India?" },
  { icon: "🍂", short: "Yellow leaf diagnosis", full: "My rice crop leaves are turning yellow at the tips. What could be wrong and how do I fix it?" },
  { icon: "💧", short: "Irrigation for rice", full: "What is the best irrigation schedule and method for growing rice in a 5-acre field?" },
  { icon: "🐄", short: "Increase milk production", full: "How can I increase the daily milk yield of my Holstein-Friesian cattle herd?" },
  { icon: "🌿", short: "Tomato fertilizer plan", full: "What is the complete fertilizer and nutrient schedule for growing tomatoes from seedling to harvest?" },
  { icon: "🐛", short: "Pest control guide", full: "What are the most effective organic and chemical pest control methods for cotton crops?" },
];

// ─── Main ChatCanvas ────────────────────────────────────────────────────────
export default function ChatCanvas({
  messages,
  onSuggestionClick,
}: {
  messages: Message[];
  onSuggestionClick: (text: string) => void;
}) {
  const bottomRef = useRef<HTMLDivElement>(null);
  const scrollContainerRef = useRef<HTMLDivElement>(null);
  
  // Memoize the last message to prevent excessive scrolling
  const lastMessage = useMemo(() => {
    if (messages.length === 0) return null;
    return messages[messages.length - 1];
  }, [messages]);

  // Optimized auto-scroll - only scroll when there's a new message or content change
  useEffect(() => {
    const scrollToBottom = () => {
      if (bottomRef.current && scrollContainerRef.current) {
        const container = scrollContainerRef.current;
        const isNearBottom = container.scrollHeight - container.scrollTop - container.clientHeight < 100;
        
        // Only auto-scroll if user is near bottom or it's a new message
        if (isNearBottom || messages.length <= 1) {
          bottomRef.current.scrollIntoView({ behavior: "smooth", block: "end" });
        }
      }
    };
    
    // Use requestAnimationFrame to avoid layout thrashing
    const animationId = requestAnimationFrame(scrollToBottom);
    return () => cancelAnimationFrame(animationId);
  }, [lastMessage?.id, lastMessage?.content, messages.length]);

  return (
    <div ref={scrollContainerRef} className="flex-1 overflow-y-auto w-full pb-36" style={{ scrollbarWidth: "thin" }}>
      <div className="max-w-3xl mx-auto flex flex-col gap-5 px-4 pt-6 pb-4">

        {/* ── Empty state ── */}
        {messages.length === 0 && (
          <div className="flex flex-col items-center text-center mt-12 select-none">
            {/* Logo */}
            <div
              className="w-14 h-14 rounded-2xl flex items-center justify-center mb-5 text-2xl"
              style={{ background: "rgba(22,163,74,0.12)", border: "1px solid rgba(22,163,74,0.25)" }}
            >
              🌾
            </div>

            <h1 className="text-2xl font-semibold mb-2" style={{ color: "#f2f2f2" }}>
              Farm360 AI Expert
            </h1>
            <p className="text-sm mb-8 max-w-xs leading-relaxed" style={{ color: "#666" }}>
              Your intelligent agricultural advisor. Ask anything about crops, diseases, soil, irrigation, or livestock — in English or Hindi.
            </p>

            {/* Suggestion grid */}
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-2 w-full max-w-xl text-left">
              {SUGGESTIONS.map((s) => (
                <button
                  key={s.short}
                  onClick={() => onSuggestionClick(s.full)}
                  className="chip flex items-start gap-3 px-4 py-3 rounded-xl text-left"
                  style={{
                    background: "var(--surface)",
                    border: "1px solid var(--border)",
                    color: "var(--text)",
                  }}
                >
                  <span className="text-xl shrink-0">{s.icon}</span>
                  <div>
                    <div className="text-sm font-medium leading-snug" style={{ color: "#e0e0e0" }}>{s.short}</div>
                    <div className="text-xs mt-0.5 leading-relaxed" style={{ color: "#555" }}>{s.full.slice(0, 60)}…</div>
                  </div>
                </button>
              ))}
            </div>
          </div>
        )}

        {/* ── Messages ── */}
        {messages.map((msg) => (
          <div key={msg.id} className="msg-in">
            {msg.role === "user" ? (
              /* User bubble */
              <div className="flex justify-end gap-2.5 items-end">
                <div className="max-w-[78%] space-y-2">
                  {msg.imagePreview && (
                    <img
                      src={msg.imagePreview}
                      alt="Uploaded image"
                      className="max-h-48 rounded-xl object-contain ml-auto block"
                    />
                  )}
                  {msg.content && (
                    <div
                      className="text-sm leading-relaxed px-4 py-3 rounded-2xl rounded-br-sm text-white"
                      style={{ background: "var(--user-blue)" }}
                    >
                      {msg.content}
                    </div>
                  )}
                </div>
                <div
                  className="shrink-0 w-7 h-7 rounded-full flex items-center justify-center mb-0.5"
                  style={{ background: "var(--surface-2)", border: "1px solid var(--border-2)" }}
                >
                  <User size={13} style={{ color: "#777" }} />
                </div>
              </div>
            ) : (
              /* Assistant bubble */
              <div className="flex justify-start gap-2.5 items-end msg-group">
                <div
                  className="shrink-0 w-7 h-7 rounded-full flex items-center justify-center mb-0.5"
                  style={{ background: "rgba(22,163,74,0.15)", border: "1px solid rgba(22,163,74,0.3)" }}
                >
                  <Leaf size={13} style={{ color: "#4ade80" }} />
                </div>

                <div className="flex-1 min-w-0 space-y-1">
                  <div
                    className="px-4 py-3.5 rounded-2xl rounded-bl-sm"
                    style={{ background: "var(--surface)", border: "1px solid var(--border)" }}
                  >
                    {msg.content === "" ? (
                      <TypingDots />
                    ) : (
                      <Markdown content={msg.content} streaming={msg.streaming} />
                    )}
                  </div>

                  {/* Action bar (shown after response completes) */}
                  {!msg.streaming && msg.content && (
                    <div className="flex items-center gap-0.5 pl-1">
                      <CopyButton text={msg.content} />
                      <span className="text-[11px] px-2" style={{ color: "#3a3a3a" }}>
                        {new Date(msg.timestamp).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}
                      </span>
                    </div>
                  )}
                </div>
              </div>
            )}
          </div>
        ))}

        <div ref={bottomRef} />
      </div>
    </div>
  );
}
