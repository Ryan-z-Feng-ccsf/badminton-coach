'use client'

import React, { useState, useRef } from 'react';

interface CoachFeedback {
  problem: string;
  improvement: string;
  power_technique: string;
}

export default function Home() {
  const [file, setFile] = useState<File | null>(null);
  const [action, setAction] = useState<string>("");
  const [isLoading, setLoading] = useState<boolean>(false);
  const [hasError, setError] = useState<boolean>(false);
  const [feedback, setFeedback] = useState<CoachFeedback | null>(null);
  const [isDragging, setIsDragging] = useState<boolean>(false);
  const [showModal, setShowModal] = useState(false);
  
  // 新增：用于关联点击上传的文件输入框
  const fileInputRef = useRef<HTMLInputElement>(null);

  const techniques = [
    { id: 'high_clear', label: 'High Clear' },
    {
      id: 'smash',
      label: 'Smash ➔',
      children: [
        { id: 'smash_standard', label: 'Standard Smash' },
        { id: 'smash_stick', label: 'Stick Smash' },
        { id: 'smash_jump', label: 'Jump Smash' },
        { id: 'smash_slice', label: 'Slice Smash' },
      ]
    },
    { id: 'half_smash', label: 'Half Smash' },
    {
      id: 'drop_shot',
      label: 'Drop Shot ➔',
      children: [
        { id: 'drop_slice', label: 'Slice Drop' },
        { id: 'drop_reverse_slice', label: 'Reverse Slice Drop' },
      ]
    },
    { id: 'net_shot', label: 'Net Shot' },
    { id: 'net_spin', label: 'Net Spin' },
  ];

  const handleDragOver = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragging(true);
  }

  const handleDragLeave = (e: React.DragEvent<HTMLElement>) => {
    e.preventDefault();
    setIsDragging(false);
  }

  const handleDrop = (e: React.DragEvent<HTMLElement>) => {
    e.preventDefault();
    setIsDragging(false);
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      setFile(e.dataTransfer.files[0]);
    }
  }

  // 新增：处理常规点击选择文件
  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      setFile(e.target.files[0]);
    }
  }

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!file) {
      alert("Please select a video file first");
      return;
    }

    const formData = new FormData();
    formData.append("action", action);
    formData.append("video", file);

    // 重置状态
    setError(false);
    setLoading(true);
    setFeedback(null); // 清除旧反馈

    try {
      const response = await fetch(
        'http://127.0.0.1:8001/upload-video',
        {
          method: 'POST',
          body: formData
        }
      );
      
      const result = await response.json();
      console.log("Result from the backend", result);

      // 修复：移除冗余的嵌套 if
      if (result.status === 'processed') {
        setFeedback(result.llm_feedback);
        setLoading(false);
        setFile(null); 
        setShowModal(true); 
      }
    } catch (error) {
      console.log("Network request failed", error);
      setError(true);
      setLoading(false);
      alert('There is an unexpected error');
    }
  }

  return (
    <main className='min-h-screen p-8 flex flex-col items-center justify-center bg-gray-50'>
      <h1 className='text-3xl font-bold mb-8 text-center text-gray-800'>Badminton AI - Video</h1>
      <form onSubmit={handleSubmit} className='flex flex-col gap-6 w-full max-w-md bg-white p-6 rounded-xl shadow-sm border border-gray-100'>

        {/* 下拉菜单 */}
        <div className="relative group">
          <div className='border border-gray-300 rounded-lg p-3 bg-white text-gray-700 cursor-default flex justify-between items-center group-hover:ring-2 group-hover:ring-blue-500 transition-all'>
            <span>{techniques.flatMap(t => t.children ? t.children : t).find(t => t.id === action)?.label || 'Select a technique'}</span>
            <span className="text-gray-400 text-xs">▼</span>
          </div>

          <ul className='absolute left-0 mt-1 w-full bg-white border border-gray-200 rounded-lg shadow-xl opacity-0 invisible group-hover:opacity-100 group-hover:visible transition-all duration-200 z-50 py-1'>
            {techniques.map((item) => (
              <li key={item.id} className="relative group/item px-4 py-2 hover:bg-blue-50 cursor-pointer text-gray-700 flex justify-between items-center">
                {!item.children ? (
                  <div className="w-full" onClick={() => setAction(item.id)}>{item.label}</div>
                ) : (
                  <>
                    <span>{item.label}</span>
                    <ul className='absolute left-full top-0 ml-1 w-48 bg-white border border-gray-200 rounded-lg shadow-xl opacity-0 invisible group-hover/item:opacity-100 group-hover/item:visible transition-all duration-200 py-1'>
                      {item.children.map((child) => (
                        <li
                          key={child.id}
                          className="px-4 py-2 hover:bg-blue-600 hover:text-white transition-colors"
                          onClick={() => setAction(child.id)}
                        >
                          {child.label}
                        </li>
                      ))}
                    </ul>
                  </>
                )}
              </li>
            ))}
          </ul>
        </div>

        {/* 拖拽/点击上传区域 */}
        <div
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onDrop={handleDrop}
          onClick={() => fileInputRef.current?.click()} // 新增：点击触发文件选择
          className={`
            border-2 border-dashed rounded-xl p-10 text-center transition-all duration-200 cursor-pointer
            ${isDragging
              ? 'border-red-500 bg-red-50 scale-[1.02]'
              : 'border-gray-300 bg-gray-50 hover:border-blue-400'}
          `}
        >
          {/* 隐藏的文件输入框 */}
          <input 
            type="file" 
            accept="video/*" 
            className="hidden" 
            ref={fileInputRef} 
            onChange={handleFileSelect} 
          />

          {file ? (
            <p className="text-green-600 font-semibold flex items-center justify-center gap-2">
              <span>✅</span> {file.name}
            </p>
          ) : (
            <div className={isDragging ? "text-red-500 font-bold" : "text-gray-500"}>
              <p className="text-lg">{isDragging ? "Drop to upload!" : "Drag & drop video here"}</p>
              <p className="text-sm mt-1 opacity-70">or click to browse</p>
            </div>
          )}
        </div>

        <button 
          type='submit' 
          disabled={isLoading}
          className={`text-white p-3 rounded-lg font-bold transition-all shadow-md active:scale-[0.98] ${
            isLoading ? 'bg-blue-400 cursor-not-allowed' : 'bg-blue-600 hover:bg-blue-700'
          }`}
        >
          {isLoading ? 'Uploading...' : 'Upload & Analyze'}
        </button>

        {/* 加载动画 */}
        {isLoading && (
          <div className="flex flex-col items-center gap-3 py-2">
            <span className="text-gray-600 font-medium text-sm">🏸 Analyzing, please wait...</span>
            <svg viewBox="0 0 100 100" className="w-16 h-16">
              <style>
                {`
                  @keyframes spin-shuttle {
                    from { transform: rotate(0deg); }
                    to { transform: rotate(360deg); }
                  }
                `}
              </style>
              <defs>
                <linearGradient id="beltGradient" x1="0%" y1="0%" x2="100%" y2="0%">
                  <stop offset="0%" stopColor="#064e3b" />
                  <stop offset="50%" stopColor="#047857" />
                  <stop offset="100%" stopColor="#064e3b" />
                </linearGradient>
              </defs>
              <circle cx="50" cy="85" r="10" fill="#cbd5e1" stroke="#64748b" strokeWidth="1" />
              <g style={{ animation: 'spin-shuttle 1.5s linear infinite', transformOrigin: '50% 65%' }}>
                <rect x="38" y="75" width="24" height="4" fill="url(#beltGradient)" stroke="#022c22" strokeWidth="0.5" />
                {[...Array(16)].map((_, i) => (
                  <g key={i} transform={`rotate(${i * (360 / 16)} 50 65)`}>
                    <rect x="49.5" y="73" width="1" height="15" fill="#94a3b8" stroke="#64748b" strokeWidth="0.5" />
                    <path d="M47,65 C46,55 46,45 47,35 C48,25 52,25 53,35 C54,45 54,55 53,65 L50,73 Z" fill="#e2e8f0" stroke="#94a3b8" strokeWidth="0.5" />
                    <line x1="50" y1="35" x2="50" y2="73" stroke="#94a3b8" strokeWidth="0.5" />
                  </g>
                ))}
              </g>
            </svg>
          </div>
        )}

        {hasError && (
          <div className='p-4 bg-red-50 border border-red-200 rounded-lg text-red-600 text-center text-sm'>
            ❌ Network Request Error, please try again
          </div>
        )}
      </form>

      {/* 模态框 (Modal) */}
      {showModal && feedback && (
        <div className="fixed inset-0 z-[100] flex items-center justify-center p-4 sm:p-6">
          <div
            className="absolute inset-0 bg-black/60 backdrop-blur-sm"
            onClick={() => setShowModal(false)}
          />

          <div className="relative bg-white w-full max-w-2xl max-h-[90vh] overflow-y-auto rounded-2xl shadow-2xl flex flex-col animate-in fade-in zoom-in duration-300">
            <div className="sticky top-0 bg-white border-b border-gray-100 px-6 py-4 flex justify-between items-center">
              <h2 className="text-xl font-bold text-gray-800 flex items-center gap-2">
                <span className="text-2xl">🏸</span> AI Coach Analysis
              </h2>
              <button
                onClick={() => setShowModal(false)}
                className="p-2 hover:bg-gray-100 rounded-full transition-colors text-gray-400 hover:text-gray-600"
              >
                <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M6 18L18 6M6 6l12 12" />
                </svg>
              </button>
            </div>

            <div className="p-6 space-y-8">
              <section>
                <div className="flex items-center gap-2 mb-3">
                  <div className="w-1 h-6 bg-red-500 rounded-full" />
                  <h3 className="font-bold text-gray-700 uppercase tracking-wider text-sm">Biomechanical Issue</h3>
                </div>
                <div className="bg-red-50 border border-red-100 rounded-xl p-4 text-gray-800 leading-relaxed">
                  {feedback.problem}
                </div>
              </section>

              <section>
                <div className="flex items-center gap-2 mb-3">
                  <div className="w-1 h-6 bg-blue-500 rounded-full" />
                  <h3 className="font-bold text-gray-700 uppercase tracking-wider text-sm">Action Plan</h3>
                </div>
                <div className="bg-blue-50 border border-blue-100 rounded-xl p-4 text-gray-800 leading-relaxed">
                  {feedback.improvement}
                </div>
              </section>

              <section>
                <div className="flex items-center gap-2 mb-3">
                  <div className="w-1 h-6 bg-amber-500 rounded-full" />
                  <h3 className="font-bold text-gray-700 uppercase tracking-wider text-sm">Power Generation</h3>
                </div>
                <div className="bg-amber-50 border border-amber-100 rounded-xl p-4 text-gray-800 leading-relaxed italic shadow-sm">
                  “{feedback.power_technique}”
                </div>
              </section>
            </div>

            <div className="p-6 border-t border-gray-50 text-right">
              <button
                onClick={() => setShowModal(false)}
                className="bg-gray-900 text-white px-6 py-2 rounded-lg font-medium hover:bg-gray-800 transition-all active:scale-95"
              >
                Got it, Coach!
              </button>
            </div>
          </div>
        </div>
      )}
    </main>
  )
}