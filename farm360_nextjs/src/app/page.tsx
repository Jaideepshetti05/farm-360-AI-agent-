"use client";
import React, { useState } from "react";
import Sidebar from "@/components/Sidebar";
import ChatCanvas from "@/components/ChatCanvas";
import ChatInput from "@/components/ChatInput";

type Message = {
  role: "user" | "assistant";
  content: string;
  imagePreview?: string;
  streaming?: boolean;
};

export default function Home() {
  const [messages, setMessages] = useState<Message[]>([]);

  return (
    <div className="flex bg-gray-900 text-white h-screen w-screen overflow-hidden">
      <Sidebar
        onNewChat={() => setMessages([])}
        onClearChat={() => setMessages([])}
      />
      <div className="flex-1 flex flex-col relative h-full">
        <ChatCanvas messages={messages} />
        <ChatInput setMessages={setMessages as any} />
      </div>
    </div>
  );
}
