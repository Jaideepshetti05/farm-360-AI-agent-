"use client";
import React from "react";

// ─── Types ───────────────────────────────────────────────────────────────────
interface StructuredResponse {
  summary?: string;
  analysis?: string;
  recommendations?: string[];
  crop_suggestions?: string[];
  next_steps?: string[];
}

// ─── Reusable Section Card ────────────────────────────────────────────────────
function SectionCard({
  icon,
  title,
  accent,
  children,
}: {
  icon: string;
  title: string;
  accent: string; // tailwind border+bg classes
  children: React.ReactNode;
}) {
  return (
    <div className={`rounded-xl border p-4 ${accent}`}>
      <p className="text-[11px] font-semibold uppercase tracking-widest mb-2.5 opacity-60 flex items-center gap-1.5">
        <span className="text-base leading-none">{icon}</span>
        {title}
      </p>
      <div className="text-sm leading-relaxed text-gray-200">{children}</div>
    </div>
  );
}

// ─── Bullet List ─────────────────────────────────────────────────────────────
function BulletList({ items }: { items: string[] }) {
  return (
    <ul className="space-y-2">
      {items.map((item, i) => (
        <li key={i} className="flex items-start gap-2.5">
          <span className="shrink-0 mt-[5px] w-1.5 h-1.5 rounded-full bg-green-400" />
          <span>{item}</span>
        </li>
      ))}
    </ul>
  );
}

// ─── Numbered Step List ───────────────────────────────────────────────────────
function NumberedList({ items }: { items: string[] }) {
  return (
    <ol className="space-y-2">
      {items.map((item, i) => (
        <li key={i} className="flex items-start gap-2.5">
          <span className="shrink-0 mt-0.5 w-5 h-5 rounded-full flex items-center justify-center text-[10px] font-bold bg-purple-700 text-purple-100">
            {i + 1}
          </span>
          <span>{item}</span>
        </li>
      ))}
    </ol>
  );
}

// ─── Structured Response Card ─────────────────────────────────────────────────
function StructuredCard({ data }: { data: StructuredResponse }) {
  return (
    <div className="w-full space-y-2.5">
      {/* Summary — banner */}
      {data.summary && (
        <div className="bg-gradient-to-r from-green-900/50 to-emerald-900/40 border border-green-700/50 rounded-xl px-4 py-3">
          <p className="text-green-300 font-semibold text-sm leading-snug">
            🌾 {data.summary}
          </p>
        </div>
      )}

      {/* Analysis */}
      {data.analysis && (
        <SectionCard
          icon="🔍"
          title="Analysis"
          accent="border-blue-800/50 bg-blue-950/25"
        >
          <p>{data.analysis}</p>
        </SectionCard>
      )}

      {/* Recommendations */}
      {data.recommendations && data.recommendations.length > 0 && (
        <SectionCard
          icon="✅"
          title="Recommendations"
          accent="border-emerald-800/50 bg-emerald-950/20"
        >
          <BulletList items={data.recommendations} />
        </SectionCard>
      )}

      {/* Crop Suggestions */}
      {data.crop_suggestions && data.crop_suggestions.length > 0 && (
        <SectionCard
          icon="🌱"
          title="Crop Suggestions"
          accent="border-yellow-800/40 bg-yellow-950/15"
        >
          <div className="flex flex-wrap gap-2 pt-0.5">
            {data.crop_suggestions.map((crop, i) => (
              <span
                key={i}
                className="px-3 py-1 rounded-full text-xs font-medium bg-yellow-900/50 border border-yellow-700/60 text-yellow-200"
              >
                {crop}
              </span>
            ))}
          </div>
        </SectionCard>
      )}

      {/* Next Steps */}
      {data.next_steps && data.next_steps.length > 0 && (
        <SectionCard
          icon="📌"
          title="Next Steps"
          accent="border-purple-800/50 bg-purple-950/20"
        >
          <NumberedList items={data.next_steps} />
        </SectionCard>
      )}
    </div>
  );
}

// ─── Assistant Message — parses JSON or falls back to plain text ──────────────
function AssistantMessage({ content, streaming }: { content: string; streaming?: boolean }) {
  // Show pulsing dots while waiting for first content
  if (content === "") {
    return (
      <div className="flex gap-1.5 items-center py-2 px-1">
        <span className="w-2 h-2 rounded-full bg-green-400 animate-pulse" />
        <span className="w-2 h-2 rounded-full bg-green-400 animate-pulse [animation-delay:150ms]" />
        <span className="w-2 h-2 rounded-full bg-green-400 animate-pulse [animation-delay:300ms]" />
      </div>
    );
  }

  // While actively streaming, show a "thinking" indicator instead of partial JSON
  if (streaming) {
    return (
      <div className="flex items-center gap-2 text-sm text-gray-400 py-1">
        <span className="w-2 h-2 rounded-full bg-green-500 animate-pulse" />
        Analyzing your query…
      </div>
    );
  }

  // Try to parse as structured JSON once stream is complete
  let structured: StructuredResponse | null = null;
  try {
    const parsed = JSON.parse(content.trim());
    if (
      parsed &&
      typeof parsed === "object" &&
      !Array.isArray(parsed) &&
      (parsed.summary || parsed.analysis || parsed.recommendations)
    ) {
      structured = parsed as StructuredResponse;
    }
  } catch {
    // Not JSON — fall through to plain text
  }

  if (structured) {
    return <StructuredCard data={structured} />;
  }

  // Fallback: plain text (error messages, legacy responses)
  return (
    <p className="text-sm text-gray-200 leading-relaxed whitespace-pre-wrap">
      {content}
    </p>
  );
}

// ─── Suggestion Chips (shown on empty state) ──────────────────────────────────
const SUGGESTIONS = [
  "Suggest better crops",
  "My crop leaves are yellow",
  "Best irrigation method",
  "Dairy production forecast",
];

// ─── Main ChatCanvas ──────────────────────────────────────────────────────────
export default function ChatCanvas({
  messages,
}: {
  messages: Array<{ role: string; content: string; imagePreview?: string; streaming?: boolean }>;
}) {
  return (
    <div className="flex-1 overflow-y-auto w-full pb-36">
      <div className="max-w-3xl mx-auto flex flex-col gap-6 p-6 mt-8">

        {/* Empty state */}
        {messages.length === 0 && (
          <div className="text-center mt-24 select-none">
            <div className="inline-flex items-center justify-center w-16 h-16 rounded-2xl bg-green-900/40 border border-green-700/50 mb-5 text-3xl">
              🌾
            </div>
            <h1 className="text-3xl font-bold text-white mb-2 tracking-tight">
              Farm360 AI Expert
            </h1>
            <p className="text-gray-400 text-sm max-w-sm mx-auto leading-relaxed">
              Your intelligent agricultural advisor. Ask about crops, diseases,
              irrigation, or upload a plant image for diagnosis.
            </p>
            <div className="flex flex-wrap justify-center gap-2 mt-6">
              {SUGGESTIONS.map((hint) => (
                <span
                  key={hint}
                  className="px-3 py-1.5 rounded-full text-xs bg-gray-800 border border-gray-700 text-gray-300"
                >
                  {hint}
                </span>
              ))}
            </div>
          </div>
        )}

        {/* Messages */}
        {messages.map((msg, i) => (
          <div
            key={i}
            className={`flex items-start gap-2 ${
              msg.role === "user" ? "justify-end" : "justify-start"
            }`}
          >
            {/* Avatar for assistant */}
            {msg.role === "assistant" && (
              <div className="shrink-0 w-7 h-7 rounded-full bg-green-900 border border-green-700 flex items-center justify-center text-sm mt-1">
                🌿
              </div>
            )}

            <div
              className={`max-w-[85%] rounded-2xl ${
                msg.role === "user"
                  ? "bg-blue-600 text-white px-4 py-3"
                  : "bg-gray-800/80 border border-gray-700/60 px-4 py-4"
              }`}
            >
              {/* Image preview for user uploads */}
              {msg.imagePreview && (
                <img
                  src={msg.imagePreview}
                  alt="upload preview"
                  className="max-h-64 object-contain rounded-lg mb-3"
                />
              )}

              {msg.role === "assistant" ? (
                <AssistantMessage
                  content={msg.content}
                  streaming={msg.streaming}
                />
              ) : (
                <p className="text-sm leading-relaxed">{msg.content}</p>
              )}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
