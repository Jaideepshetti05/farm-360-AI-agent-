import React from "react";
import { Plus, Trash2 } from "lucide-react";

export default function Sidebar({ onNewChat, onClearChat }: { onNewChat: any, onClearChat: any }) {
  return (
    <div className="w-[260px] h-full bg-black flex flex-col p-3 border-r border-gray-800 hidden md:flex">
      <button 
        onClick={onNewChat} 
        className="flex gap-3 items-center text-white bg-transparent hover:bg-gray-800 p-3 rounded-md transition-colors border border-gray-700"
      >
        <Plus size={16} /> New Chat
      </button>

      <div className="flex-1 mt-6">
        <div className="text-xs text-gray-400 mb-3 px-2 font-semibold">Today</div>
        <div className="px-2 py-2 text-sm text-gray-300 hover:bg-gray-800 rounded-md cursor-pointer truncate">
          Active Conversation
        </div>
      </div>

      <div className="mt-auto border-t border-gray-800 pt-3">
        <button 
          onClick={onClearChat} 
          className="flex gap-3 items-center w-full text-red-400 hover:bg-gray-900 p-3 rounded-md transition-colors"
        >
          <Trash2 size={16} /> Clear Conversations
        </button>
      </div>
    </div>
  );
}
