"use client";
import React, { useState, useRef, useEffect } from "react";
import { Send, Paperclip } from "lucide-react";

export default function ChatInput({ setMessages }: { setMessages: any }) {
  const [query, setQuery] = useState("");
  const [loading, setLoading] = useState(false);
  const [file, setFile] = useState<File | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const bottomRef = useRef<HTMLDivElement>(null);
  // Accumulator ref: avoids React StrictMode double-invoke duplicating streamed lines
  const contentAccRef = useRef("");

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [loading]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!query.trim() && !file) return;

    // Create a local blob URL preview if there's a file
    const previewUrl = file ? URL.createObjectURL(file) : undefined;
    
    // Add User message
    const userMessage = { role: "user", content: query, imagePreview: previewUrl };
    setMessages((prev: any) => [...prev, userMessage]);
    
    // Reset inputs
    setQuery("");
    const uploadRef = file;
    setFile(null);
    setLoading(true);

    // Add empty Assistant placeholder for streaming
    setMessages((prev: any) => [...prev, { role: "assistant", content: "" }]);

    try {
      const formData = new FormData();
      if (query.trim()) formData.append("query", query);
      formData.append("session_id", "default_react_session");
      if (uploadRef) formData.append("image", uploadRef);

      const baseUrl = process.env.NEXT_PUBLIC_API_BASE_URL || "http://127.0.0.1:8000";
      const apiKey = process.env.NEXT_PUBLIC_FARM360_API_KEY || "secure-secret-key-1234";
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

      // If it's an image, the backend currently does not stream (it returns JSON)
      if (uploadRef) {
          const resp = await response.json();
          setMessages((prev: any) => {
             const updated = [...prev];
             updated[updated.length - 1].content = resp.response || JSON.stringify(resp);
             return updated;
          });
          setLoading(false);
          return;
      }

      // Bulletproof SSE reader - accumulates content in a ref, pushes to React once via rAF
      // This avoids React 18 StrictMode double-setState duplication
      const reader = response.body.getReader();
      const decoder = new TextDecoder("utf-8");
      let buffer = "";
      contentAccRef.current = "";

      // Throttled flusher: update UI every animation frame
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
            contentAccRef.current += data;
            if (!rafId) rafId = requestAnimationFrame(flush);
          }
        }
      }
      // Final flush for any remaining content
      flush();
    } catch (e: any) {
       console.error(e);
       setMessages((prev: any) => {
          const updated = [...prev];
           updated[updated.length - 1].content = `⚠️ System Error: ${e.message}`;
           return updated;
       });
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="absolute bottom-0 w-full p-4 bg-gradient-to-t from-gray-900 via-gray-900 to-transparent">
      
      {/* File Preview Attachment */}
      {file && (
        <div className="max-w-3xl mx-auto mb-2 p-2 bg-gray-800 rounded-lg inline-flex items-center gap-2 border border-gray-700">
           <span className="text-xs text-gray-300">📎 {file.name} attached</span>
           <button onClick={() => setFile(null)} className="text-red-400 font-bold ml-2 hover:text-red-300">x</button>
        </div>
      )}

      {/* Main Input Form */}
      <form onSubmit={handleSubmit} className="max-w-3xl mx-auto relative flex items-center bg-gray-800 rounded-2xl p-2 md:p-3 border border-gray-700 shadow-2xl focus-within:border-gray-500 transition-colors">
        
        <input 
          type="file" 
          accept="image/*" 
          className="hidden" 
          ref={fileInputRef} 
          onChange={(e) => {
             if (e.target.files && e.target.files[0]) {
                setFile(e.target.files[0]);
             }
          }}
        />
        
        <button 
          type="button" 
          onClick={() => fileInputRef.current?.click()}
          className="p-2 text-gray-400 hover:text-white rounded-full hover:bg-gray-700"
        >
          <Paperclip size={20}/>
        </button>

        <textarea 
          rows={1}
          value={query} 
          onChange={(e) => setQuery(e.target.value)} 
          onKeyDown={(e) => {
             if (e.key === 'Enter' && !e.shiftKey) { 
               e.preventDefault(); 
               handleSubmit(e); 
             }
          }}
          placeholder={loading ? "Farm360 is typing..." : "Message Farm360 Agent..."}
          disabled={loading}
          className="flex-1 bg-transparent border-none focus:outline-none text-white px-3 py-1 resize-none max-h-40 overflow-y-auto"
        />
        
        <button 
          type="submit" 
          disabled={loading || (!query.trim() && !file)} 
          className="p-2 ml-1 bg-white rounded-full text-black disabled:opacity-30 hover:bg-gray-200 transition-all"
        >
          <Send size={18} fill="currentColor"/>
        </button>

      </form>
      <div className="text-center text-xs text-gray-500 mt-3 hidden md:block">
        Farm360 AI can make mistakes. Consider verifying crop suggestions and veterinary info.
      </div>
    </div>
  );
}
