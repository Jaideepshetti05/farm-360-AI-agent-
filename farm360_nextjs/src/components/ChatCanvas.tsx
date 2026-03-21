"use client";
import React from "react";
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';

export default function ChatCanvas({ messages }: { messages: any[] }) {
  return (
    <div className="flex-1 overflow-y-auto w-full pb-32">
      <div className="max-w-3xl mx-auto flex flex-col gap-6 p-6 mt-10">
        
        {messages.length === 0 && (
           <div className="text-center text-gray-400 mt-20">
             <h1 className="text-3xl font-bold mb-4">Farm360 Digital Assistant</h1>
             <p className="opacity-80">How can I help your farm today? Predict yields, analyze crops, and forecast trends.</p>
           </div>
        )}

        {messages.map((msg, i) => (
          <div key={i} className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
            <div className={`p-4 max-w-[85%] rounded-2xl ${
              msg.role === 'user' 
                ? 'bg-blue-600 text-white' 
                : 'bg-gray-800 text-gray-100 border border-gray-700'
            }`}>
              
              {/* If an image preview is attached to a user message, show it inline */}
              {msg.imagePreview && (
                <img src={msg.imagePreview} alt="upload" className="max-h-64 object-contain rounded-md mb-2" />
              )}
              
              {msg.role === 'assistant' ? (
                <article className="prose prose-invert prose-p:leading-relaxed max-w-none text-sm md:text-base">
                  <ReactMarkdown remarkPlugins={[remarkGfm]}>{msg.content}</ReactMarkdown>
                  {/* Blinking loader if waiting for first byte */}
                  {msg.content === "" && (
                    <div className="flex gap-1 items-center opacity-70 mt-1">
                      <span className="w-2 h-2 rounded-full bg-white animate-pulse"></span>
                      <span className="w-2 h-2 rounded-full bg-white animate-pulse delay-75"></span>
                      <span className="w-2 h-2 rounded-full bg-white animate-pulse delay-150"></span>
                    </div>
                  )}
                </article>
              ) : (
                msg.content
              )}
              
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
