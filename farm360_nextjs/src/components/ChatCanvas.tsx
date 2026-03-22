"use client";
import React from "react";

// --- Type for structured AI responses ---
interface StructuredResponse {
  summary?: string;
  analysis?: string;
  recommendations?: string[];
  crop_suggestions?: string[];
  next_steps?: string[];
}

// --- Sub-components for structured rendering ---

const SectionCard = ({ icon, title, children, color }: { icon: string; title: string; children: React.ReactNode; color: string }) => (
  <div className={`rounded-xl border ${color} p-4 mb-3`}>
    <h3 className="text-sm font-semibold mb-2 flex items-center gap-2 uppercase tracking-wider opacity-80">
      <span>{icon}</span> {title}
    </h3>
    <div className="text-sm text-gray-200 leading-relaxed">{children}</div>
  </div>
);

const BulletList = ({ items }: { items: string[] }) => (
  <ul className="space-y-1.5">
    {items.map((item, i) => (
      <li key={i} className="flex items-start gap-2">
        <span className="mt-1 w-1.5 h-1.5 rounded-full bg-green-400 shrink-0" />
        <span>{item}</span>
      </li>
    ))}
  </ul>
);

function StructuredCard({ data }: { data: StructuredResponse }) {
  return (
    <div className="space-y-1 w-full">
      {/* Summary Banner */}
      {data.summary && (
        <div className="bg-gradient-to-r from-green-900/60 to-emerald-900/60 border border-green-700/50 rounded-xl px-4 py-3 mb-3">
          <p className="text-green-300 font-medium text-sm leading-snug">🌾 {data.summary}</p>
        </div>
      )}

      {/* Analysis */}
      {data.analysis && (
        <SectionCard icon="🔬" title="Analysis" color="border-blue-800/60 bg-blue-950/30">
          <p>{data.analysis}</p>
        </SectionCard>
      )}

      {/* Recommendations */}
      {data.recommendations && data.recommendations.length > 0 && (
        <SectionCard icon="✅" title="Recommendations" color="border-emerald-800/60 bg-emerald-950/30">
          <BulletList items={data.recommendations} />
        </SectionCard>
      )}

      {/* Crop Suggestions */}
      {data.crop_suggestions && data.crop_suggestions.length > 0 && (
        <SectionCard icon="🌱" title="Crop Suggestions" color="border-yellow-800/60 bg-yellow-950/20">
          <div className="flex flex-wrap gap-2 pt-1">
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
        <SectionCard icon="🚀" title="Next Steps" color="border-purple-800/60 bg-purple-950/30">
          <ol className="space-y-1.5">
            {data.next_steps.map((step, i) => (
              <li key={i} className="flex items-start gap-2">
                <span className="shrink-0 w-5 h-5 rounded-full bg-purple-700 text-purple-100 text-xs flex items-center justify-center font-bold mt-0.5">{i + 1}</span>
                <span>{step}</span>
              </li>
            ))}
          </ol>
        </SectionCard>
      )}
    </div>
  );
}

function AssistantMessage({ content }: { content: string }) {
  // Try to parse as structured JSON
  if (content === "") {
    return (
      <div className="flex gap-1 items-center opacity-70 mt-1 p-2">
        <span className="w-2 h-2 rounded-full bg-green-400 animate-pulse" />
        <span className="w-2 h-2 rounded-full bg-green-400 animate-pulse delay-75" />
        <span className="w-2 h-2 rounded-full bg-green-400 animate-pulse delay-150" />
      </div>
    );
  }

  let structured: StructuredResponse | null = null;
  try {
    const parsed = JSON.parse(content);
    if (parsed && typeof parsed === "object" && (parsed.summary || parsed.analysis)) {
      structured = parsed as StructuredResponse;
    }
  } catch {
    // Not JSON, fall through to plain text
  }

  if (structured) {
    return <StructuredCard data={structured} />;
  }

  // Fallback: render as plain text (legacy or error)
  return <p className="text-sm text-gray-200 leading-relaxed whitespace-pre-wrap">{content}</p>;
}

export default function ChatCanvas({ messages }: { messages: any[] }) {
  return (
    <div className="flex-1 overflow-y-auto w-full pb-36">
      <div className="max-w-3xl mx-auto flex flex-col gap-6 p-6 mt-8">

        {messages.length === 0 && (
          <div className="text-center mt-24 select-none">
            <div className="inline-flex items-center justify-center w-16 h-16 rounded-2xl bg-green-900/40 border border-green-700/50 mb-4">
              <span className="text-3xl">🌾</span>
            </div>
            <h1 className="text-3xl font-bold text-white mb-2">Farm360 AI Expert</h1>
            <p className="text-gray-400 text-sm max-w-sm mx-auto">
              Your intelligent agricultural advisor. Ask about crop yields, diseases, irrigation, or upload a crop image.
            </p>
            <div className="flex flex-wrap justify-center gap-2 mt-6">
              {["Suggest better crops", "My crop leaves are yellow", "Best irrigation method", "Dairy production forecast"].map((hint) => (
                <span key={hint} className="px-3 py-1.5 rounded-full text-xs bg-gray-800 border border-gray-700 text-gray-300 cursor-default">
                  {hint}
                </span>
              ))}
            </div>
          </div>
        )}

        {messages.map((msg, i) => (
          <div key={i} className={`flex ${msg.role === "user" ? "justify-end" : "justify-start"}`}>
            {msg.role === "assistant" && (
              <div className="w-7 h-7 rounded-full bg-green-800 border border-green-600 flex items-center justify-center text-sm shrink-0 mr-2 mt-1">
                🌿
              </div>
            )}

            <div
              className={`w-full max-w-[85%] rounded-2xl ${
                msg.role === "user"
                  ? "bg-blue-600 text-white p-4"
                  : "bg-gray-800/80 border border-gray-700/60 p-4"
              }`}
            >
              {msg.imagePreview && (
                <img
                  src={msg.imagePreview}
                  alt="upload"
                  className="max-h-64 object-contain rounded-md mb-3"
                />
              )}

              {msg.role === "assistant" ? (
                <AssistantMessage content={msg.content} />
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
