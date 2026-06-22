"use client";
import React, { useState, useRef, useEffect } from "react";
import {
  Plus, Trash2, Sprout, MessageSquare,
  Settings2, Zap, Bot, Pencil, Check, X, Search
} from "lucide-react";
import type { Conversation } from "@/app/page";

const MODELS = [
  { id: "google/gemma-4-26b-a4b-it:free",             label: "Gemma 4 26B",   tag: "FREE", icon: "🟢" },
  { id: "meta-llama/llama-3.3-70b-instruct:free",     label: "Llama 3.3 70B", tag: "FREE", icon: "🦙" },
  { id: "meta-llama/llama-3.1-8b-instruct:free",      label: "Llama 3.1 8B",  tag: "FAST", icon: "⚡" },
  { id: "mistralai/mistral-7b-instruct:free",          label: "Mistral 7B",    tag: "FREE", icon: "🌊" },
  { id: "qwen/qwen-2.5-72b-instruct:free",            label: "Qwen 2.5 72B",  tag: "FREE", icon: "🧠" },
];

interface SidebarProps {
  onNewChat: () => void;
  onClearChat: () => void;
  conversations: Conversation[];
  activeConversationId: string;
  onSelectConversation: (id: string) => void;
  onDeleteConversation: (id: string) => void;
  onRenameConversation: (id: string, newTitle: string) => void;
  selectedModel: string;
  onModelChange: (id: string) => void;
}

export default function Sidebar({
  onNewChat, onClearChat, conversations, activeConversationId,
  onSelectConversation, onDeleteConversation, onRenameConversation,
  selectedModel, onModelChange,
}: SidebarProps) {
  const [showModels, setShowModels] = useState(false);
  const [searchQuery, setSearchQuery] = useState("");
  const [renamingId, setRenamingId] = useState<string | null>(null);
  const [renameValue, setRenameValue] = useState("");
  const renameInputRef = useRef<HTMLInputElement>(null);
  const currentModel = MODELS.find(m => m.id === selectedModel) ?? MODELS[0];

  // Focus rename input when editing
  useEffect(() => {
    if (renamingId && renameInputRef.current) {
      renameInputRef.current.focus();
      renameInputRef.current.select();
    }
  }, [renamingId]);

  const filtered = searchQuery.trim()
    ? conversations.filter(c =>
        c.title.toLowerCase().includes(searchQuery.toLowerCase())
      )
    : conversations;

  const startRename = (conv: Conversation) => {
    setRenamingId(conv.id);
    setRenameValue(conv.title);
  };

  const confirmRename = () => {
    if (renamingId && renameValue.trim()) {
      onRenameConversation(renamingId, renameValue.trim());
    }
    setRenamingId(null);
  };

  // Group conversations by time
  const today = new Date();
  const groups: { label: string; items: Conversation[] }[] = [];
  const todayItems: Conversation[] = [];
  const yesterdayItems: Conversation[] = [];
  const weekItems: Conversation[] = [];
  const olderItems: Conversation[] = [];

  filtered.forEach(c => {
    const d = new Date(c.updatedAt);
    const diffDays = Math.floor((today.getTime() - d.getTime()) / 86400000);
    if (diffDays === 0) todayItems.push(c);
    else if (diffDays === 1) yesterdayItems.push(c);
    else if (diffDays <= 7) weekItems.push(c);
    else olderItems.push(c);
  });

  if (todayItems.length) groups.push({ label: "Today", items: todayItems });
  if (yesterdayItems.length) groups.push({ label: "Yesterday", items: yesterdayItems });
  if (weekItems.length) groups.push({ label: "This Week", items: weekItems });
  if (olderItems.length) groups.push({ label: "Older", items: olderItems });

  return (
    <aside
      className="w-full h-full shrink-0 flex flex-col border-r overflow-hidden"
      style={{ background: "var(--sidebar)", borderColor: "var(--border)" }}
    >
      {/* ── Logo ── */}
      <div
        className="flex items-center gap-2.5 px-4 h-14 shrink-0 border-b"
        style={{ borderColor: "var(--border)" }}
      >
        <div
          className="w-7 h-7 rounded-lg flex items-center justify-center"
          style={{ background: "rgba(22,163,74,0.15)", border: "1px solid rgba(22,163,74,0.3)" }}
        >
          <Sprout size={14} style={{ color: "#4ade80" }} />
        </div>
        <span className="font-semibold text-sm" style={{ color: "#e8e8e8" }}>Farm360 AI</span>
        <span
          className="ml-auto text-[9px] font-bold px-1.5 py-0.5 rounded-full tracking-wide"
          style={{ background: "rgba(22,163,74,0.15)", color: "#4ade80", border: "1px solid rgba(22,163,74,0.25)" }}
        >
          LIVE
        </span>
      </div>

      {/* ── New chat button ── */}
      <div className="px-3 pt-3 pb-1 shrink-0">
        <button
          onClick={onNewChat}
          className="flex items-center gap-2 w-full px-3 py-2.5 rounded-xl text-sm transition-all"
          style={{ color: "#bbb", border: "1px solid var(--border)", background: "transparent" }}
          onMouseEnter={e => { e.currentTarget.style.background = "rgba(255,255,255,0.04)"; e.currentTarget.style.color = "#fff"; }}
          onMouseLeave={e => { e.currentTarget.style.background = "transparent"; e.currentTarget.style.color = "#bbb"; }}
        >
          <Plus size={15} />
          New conversation
        </button>
      </div>

      {/* ── Search ── */}
      <div className="px-3 pt-1 pb-1 shrink-0">
        <div className="flex items-center gap-2 px-3 py-2 rounded-xl"
             style={{ background: "var(--surface)", border: "1px solid var(--border)" }}>
          <Search size={13} style={{ color: "#555", flexShrink: 0 }} />
          <input
            type="text"
            value={searchQuery}
            onChange={e => setSearchQuery(e.target.value)}
            placeholder="Search conversations…"
            className="flex-1 bg-transparent border-none outline-none text-xs"
            style={{ color: "#ccc" }}
          />
          {searchQuery && (
            <button onClick={() => setSearchQuery("")} className="hover:text-white transition-colors" style={{ color: "#555" }}>
              <X size={12} />
            </button>
          )}
        </div>
      </div>

      {/* ── Conversation list ── */}
      <div className="flex-1 px-3 py-2 overflow-y-auto" style={{ scrollbarWidth: "thin" }}>
        {groups.length === 0 && (
          <p className="text-xs px-1 mt-2" style={{ color: "#3a3a3a" }}>
            {searchQuery ? "No matching conversations." : "No conversations yet. Start chatting!"}
          </p>
        )}

        {groups.map(group => (
          <div key={group.label} className="mb-3">
            <div
              className="text-[10px] font-semibold uppercase tracking-widest mb-1.5 px-1"
              style={{ color: "#444" }}
            >
              {group.label}
            </div>

            {group.items.map(conv => (
              <div key={conv.id} className="group relative mb-0.5">
                {renamingId === conv.id ? (
                  /* Rename mode */
                  <div className="flex items-center gap-1 px-2 py-2 rounded-xl"
                       style={{ background: "rgba(255,255,255,0.05)", border: "1px solid rgba(22,163,74,0.3)" }}>
                    <input
                      ref={renameInputRef}
                      value={renameValue}
                      onChange={e => setRenameValue(e.target.value)}
                      onKeyDown={e => { if (e.key === "Enter") confirmRename(); if (e.key === "Escape") setRenamingId(null); }}
                      className="flex-1 bg-transparent border-none outline-none text-xs"
                      style={{ color: "#e0e0e0" }}
                    />
                    <button onClick={confirmRename} className="p-1 rounded hover:bg-white/10 transition-colors">
                      <Check size={12} style={{ color: "#4ade80" }} />
                    </button>
                    <button onClick={() => setRenamingId(null)} className="p-1 rounded hover:bg-white/10 transition-colors">
                      <X size={12} style={{ color: "#666" }} />
                    </button>
                  </div>
                ) : (
                  /* Normal mode */
                  <div
                    onClick={() => onSelectConversation(conv.id)}
                    role="button"
                    tabIndex={0}
                    onKeyDown={e => {
                      if (e.key === "Enter" || e.key === " ") {
                        e.preventDefault();
                        onSelectConversation(conv.id);
                      }
                    }}
                    className="flex items-center gap-2.5 w-full px-3 py-2.5 rounded-xl text-sm transition-all text-left cursor-pointer"
                    style={{
                      background: conv.id === activeConversationId ? "rgba(255,255,255,0.06)" : "transparent",
                      border: conv.id === activeConversationId ? "1px solid var(--border-2)" : "1px solid transparent",
                      color: conv.id === activeConversationId ? "#e0e0e0" : "#999",
                    }}
                    onMouseEnter={e => { if (conv.id !== activeConversationId) e.currentTarget.style.background = "rgba(255,255,255,0.03)"; }}
                    onMouseLeave={e => { if (conv.id !== activeConversationId) e.currentTarget.style.background = "transparent"; }}
                  >
                    <MessageSquare size={13} style={{ color: conv.id === activeConversationId ? "#4ade80" : "#555", flexShrink: 0 }} />
                    <span className="truncate flex-1 text-xs">{conv.title}</span>

                    {/* Hover actions */}
                    <div className="hidden group-hover:flex items-center gap-0.5 shrink-0">
                      <button
                        onClick={e => { e.stopPropagation(); startRename(conv); }}
                        className="p-1 rounded hover:bg-white/10 transition-colors"
                        title="Rename"
                      >
                        <Pencil size={11} style={{ color: "#666" }} />
                      </button>
                      <button
                        onClick={e => { e.stopPropagation(); onDeleteConversation(conv.id); }}
                        className="p-1 rounded hover:bg-red-500/10 transition-colors"
                        title="Delete"
                      >
                        <Trash2 size={11} style={{ color: "#666" }} />
                      </button>
                    </div>
                  </div>
                )}
              </div>
            ))}
          </div>
        ))}
      </div>

      {/* ── Model selector ── */}
      <div className="px-3 py-3 shrink-0" style={{ borderTop: "1px solid var(--border)" }}>
        <button
          onClick={() => setShowModels(v => !v)}
          className="flex items-center gap-2 w-full px-3 py-2.5 rounded-xl text-sm transition-all"
          style={{ background: "var(--surface)", border: "1px solid var(--border)", color: "#bbb" }}
          onMouseEnter={e => { e.currentTarget.style.borderColor = "rgba(22,163,74,0.3)"; }}
          onMouseLeave={e => { e.currentTarget.style.borderColor = "var(--border)"; }}
        >
          <Bot size={14} style={{ color: "#4ade80" }} />
          <span className="truncate flex-1 text-left">{currentModel.icon} {currentModel.label}</span>
          <span
            className="text-[9px] px-1.5 py-0.5 rounded font-bold"
            style={{ background: "rgba(22,163,74,0.12)", color: "#4ade80" }}
          >
            {currentModel.tag}
          </span>
          <Settings2 size={12} style={{ color: "#555", flexShrink: 0 }} />
        </button>

        {showModels && (
          <div
            className="mt-1.5 rounded-xl overflow-hidden"
            style={{ border: "1px solid var(--border)", background: "var(--surface)" }}
          >
            {MODELS.map(m => (
              <button
                key={m.id}
                onClick={() => { onModelChange(m.id); setShowModels(false); }}
                className="flex items-center gap-2.5 w-full px-3 py-2.5 text-sm transition-all text-left"
                style={{
                  color: m.id === selectedModel ? "#4ade80" : "#aaa",
                  background: m.id === selectedModel ? "rgba(22,163,74,0.08)" : "transparent",
                }}
                onMouseEnter={e => { if (m.id !== selectedModel) e.currentTarget.style.background = "rgba(255,255,255,0.04)"; }}
                onMouseLeave={e => { if (m.id !== selectedModel) e.currentTarget.style.background = "transparent"; }}
              >
                <span>{m.icon}</span>
                <span className="flex-1 truncate">{m.label}</span>
                <span
                  className="text-[9px] px-1.5 py-0.5 rounded font-bold"
                  style={{ background: "rgba(22,163,74,0.1)", color: "#4ade80" }}
                >
                  {m.tag}
                </span>
              </button>
            ))}
          </div>
        )}
      </div>

      {/* ── Clear current chat ── */}
      <div className="px-3 pb-4 shrink-0">
        <button
          onClick={onClearChat}
          className="flex items-center gap-2 w-full px-3 py-2 rounded-xl text-xs transition-all"
          style={{ color: "#444" }}
          onMouseEnter={e => { e.currentTarget.style.color = "#f87171"; e.currentTarget.style.background = "rgba(248,113,113,0.06)"; }}
          onMouseLeave={e => { e.currentTarget.style.color = "#444"; e.currentTarget.style.background = "transparent"; }}
        >
          <Trash2 size={13} />
          Clear conversation
        </button>
      </div>

      {/* ── Status bar ── */}
      <div
        className="px-4 py-2.5 shrink-0 flex items-center gap-2"
        style={{ borderTop: "1px solid var(--border)", background: "#0d0d0d" }}
      >
        <Zap size={11} style={{ color: "#4ade80" }} />
        <span className="text-[10px]" style={{ color: "#555" }}>
          OpenRouter · {currentModel.label}
        </span>
        <span
          className="ml-auto w-1.5 h-1.5 rounded-full"
          style={{ background: "#4ade80", boxShadow: "0 0 4px #4ade80" }}
        />
      </div>
    </aside>
  );
}
