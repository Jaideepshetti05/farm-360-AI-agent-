"use client";
import React, { useState, useRef, useCallback, useEffect } from "react";
import Sidebar from "@/components/Sidebar";
import ChatCanvas from "@/components/ChatCanvas";
import ChatInput, { ChatInputHandle } from "@/components/ChatInput";
import ErrorBoundary from "@/components/ErrorBoundary";
import { Menu } from "lucide-react";

export type Message = {
  id: string;
  role: "user" | "assistant";
  content: string;
  imagePreview?: string;
  streaming?: boolean;
  timestamp: Date;
  visionResult?: any;
};

export type Conversation = {
  id: string;
  title: string;
  messages: Message[];
  createdAt: string;
  updatedAt: string;
};

// ── LocalStorage helpers ────────────────────────────────────────────────────
const LS_KEY = "farm360_conversations";
const LS_ACTIVE_KEY = "farm360_active_conversation";

function loadConversations(): Conversation[] {
  if (typeof window === "undefined") return [];
  try {
    const raw = localStorage.getItem(LS_KEY);
    if (!raw) return [];
    const parsed = JSON.parse(raw) as Conversation[];
    // Rehydrate Date objects
    return parsed.map(c => ({
      ...c,
      messages: c.messages.map(m => ({ ...m, timestamp: new Date(m.timestamp) })),
    }));
  } catch {
    return [];
  }
}

function saveConversations(convos: Conversation[]) {
  if (typeof window === "undefined") return;
  try {
    localStorage.setItem(LS_KEY, JSON.stringify(convos));
  } catch {
    // localStorage full — silently ignore
  }
}

function loadActiveId(): string | null {
  if (typeof window === "undefined") return null;
  return localStorage.getItem(LS_ACTIVE_KEY);
}

function saveActiveId(id: string) {
  if (typeof window === "undefined") return;
  localStorage.setItem(LS_ACTIVE_KEY, id);
}

function generateId(): string {
  return `conv_${Date.now()}_${Math.random().toString(36).slice(2, 7)}`;
}

function deriveTitle(messages: Message[]): string {
  const firstUser = messages.find(m => m.role === "user");
  if (!firstUser || !firstUser.content) return "New conversation";
  const text = firstUser.content.slice(0, 50);
  return text.length < firstUser.content.length ? text + "…" : text;
}

// ── Main App ────────────────────────────────────────────────────────────────
export default function Home() {
  const [conversations, setConversations] = useState<Conversation[]>([]);
  const [activeId, setActiveId] = useState<string>("");
  const [selectedModel, setSelectedModel] = useState("google/gemma-4-26b-a4b-it:free");
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [hydrated, setHydrated] = useState(false);
  const chatInputRef = useRef<ChatInputHandle>(null);

  // ── Hydrate from localStorage on mount ──────────────────────────────────
  useEffect(() => {
    const saved = loadConversations();
    const savedActiveId = loadActiveId();

    if (saved.length > 0) {
      // eslint-disable-next-line
      setConversations(saved);
      if (savedActiveId && saved.find(c => c.id === savedActiveId)) {
        setActiveId(savedActiveId);
      } else {
        setActiveId(saved[0].id);
      }
    } else {
      // Create a fresh conversation
      const fresh: Conversation = {
        id: generateId(),
        title: "New conversation",
        messages: [],
        createdAt: new Date().toISOString(),
        updatedAt: new Date().toISOString(),
      };
      setConversations([fresh]);
      setActiveId(fresh.id);
    }
    setHydrated(true);
  }, []);

  // ── Persist to localStorage whenever conversations change ───────────────
  useEffect(() => {
    if (!hydrated) return;
    saveConversations(conversations);
  }, [conversations, hydrated]);

  useEffect(() => {
    if (!hydrated || !activeId) return;
    saveActiveId(activeId);
  }, [activeId, hydrated]);

  // ── Derive active conversation ──────────────────────────────────────────
  const activeConvo = conversations.find(c => c.id === activeId);
  const messages = activeConvo?.messages ?? [];

  // ── setMessages wrapper that updates the active conversation ────────────
  const setMessages: React.Dispatch<React.SetStateAction<Message[]>> = useCallback(
    (action) => {
      setConversations(prev => prev.map(c => {
        if (c.id !== activeId) return c;
        const newMsgs = typeof action === "function" ? action(c.messages) : action;
        const title = newMsgs.length > 0 ? deriveTitle(newMsgs) : c.title;
        return { ...c, messages: newMsgs, title, updatedAt: new Date().toISOString() };
      }));
    },
    [activeId],
  );

  // ── Handlers ────────────────────────────────────────────────────────────
  const handleNewChat = useCallback(() => {
    const fresh: Conversation = {
      id: generateId(),
      title: "New conversation",
      messages: [],
      createdAt: new Date().toISOString(),
      updatedAt: new Date().toISOString(),
    };
    setConversations(prev => [fresh, ...prev]);
    setActiveId(fresh.id);
    setSidebarOpen(false);
  }, []);

  const handleSelectConversation = useCallback((id: string) => {
    setActiveId(id);
    setSidebarOpen(false);
  }, []);

  const handleDeleteConversation = useCallback((id: string) => {
    setConversations(prev => {
      const next = prev.filter(c => c.id !== id);
      if (next.length === 0) {
        const fresh: Conversation = {
          id: generateId(),
          title: "New conversation",
          messages: [],
          createdAt: new Date().toISOString(),
          updatedAt: new Date().toISOString(),
        };
        setActiveId(fresh.id);
        return [fresh];
      }
      if (id === activeId) {
        setActiveId(next[0].id);
      }
      return next;
    });
  }, [activeId]);

  const handleRenameConversation = useCallback((id: string, newTitle: string) => {
    setConversations(prev => prev.map(c =>
      c.id === id ? { ...c, title: newTitle } : c
    ));
  }, []);

  const handleClearChat = useCallback(() => {
    setMessages([]);
  }, [setMessages]);

  const handleSuggestion = useCallback((text: string) => {
    chatInputRef.current?.sendQuery(text);
  }, []);

  // ── Don't render until hydrated (prevents flash) ────────────────────────
  if (!hydrated) {
    return (
      <div className="flex h-screen w-screen items-center justify-center" style={{ background: "var(--bg)" }}>
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 rounded-xl flex items-center justify-center text-lg"
               style={{ background: "rgba(22,163,74,0.12)", border: "1px solid rgba(22,163,74,0.25)" }}>
            🌾
          </div>
          <span className="text-sm font-medium" style={{ color: "#666" }}>Loading Farm360 AI…</span>
        </div>
      </div>
    );
  }

  return (
    <ErrorBoundary>
      <div
        className="flex h-screen w-screen overflow-hidden"
        style={{ background: "var(--bg)" }}
      >
        {/* Mobile sidebar overlay */}
        {sidebarOpen && (
          <div
            className="fixed inset-0 z-40 bg-black/50 md:hidden backdrop-blur-sm"
            onClick={() => setSidebarOpen(false)}
          />
        )}

        {/* Sidebar */}
        <div className={`
          fixed inset-y-0 left-0 z-50 w-[280px] transform transition-transform duration-300 ease-out md:relative md:translate-x-0 md:z-auto
          ${sidebarOpen ? "translate-x-0" : "-translate-x-full"}
        `}>
          <Sidebar
            onNewChat={handleNewChat}
            onClearChat={handleClearChat}
            conversations={conversations}
            activeConversationId={activeId}
            onSelectConversation={handleSelectConversation}
            onDeleteConversation={handleDeleteConversation}
            onRenameConversation={handleRenameConversation}
            selectedModel={selectedModel}
            onModelChange={setSelectedModel}
          />
        </div>

        {/* Main content area */}
        <div className="flex-1 flex flex-col relative h-full min-w-0 overflow-hidden">
          {/* Mobile header bar */}
          <div className="flex items-center h-12 px-3 shrink-0 md:hidden"
               style={{ borderBottom: "1px solid var(--border)", background: "var(--sidebar)" }}>
            <button
              onClick={() => setSidebarOpen(true)}
              className="p-2 rounded-lg transition-colors hover:bg-white/5"
              aria-label="Open sidebar"
            >
              <Menu size={18} style={{ color: "#999" }} />
            </button>
            <span className="ml-2 text-sm font-medium" style={{ color: "#ccc" }}>Farm360 AI</span>
          </div>

          <ErrorBoundary>
            <ChatCanvas
              messages={messages}
              onSuggestionClick={handleSuggestion}
            />
          </ErrorBoundary>
          <ErrorBoundary>
            <ChatInput
              ref={chatInputRef}
              messages={messages}
              setMessages={setMessages}
              selectedModel={selectedModel}
            />
          </ErrorBoundary>
        </div>
      </div>
    </ErrorBoundary>
  );
}
