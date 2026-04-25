'use client'

import React, { useState, useRef } from 'react';

interface CoachFeedback {
  problem: string;
  improvement: string;
  power_technique: string;
}

// 1. 定义多语言字典
const i18n = {
  en: {
    title: "Badminton AI - Video",
    placeholder: "Select a technique",
    dragDrop: "Drag & drop video here",
    clickBrowse: "or click to browse",
    dropUpload: "Drop to upload!",
    btnUpload: "Upload & Analyze",
    btnUploading: "Uploading...",
    analyzing: "🏸 Analyzing, please wait...",
    errorReq: "❌ Network Request Error, please try again",
    alertFile: "Please select a video file first",
    alertAction: "Please select a technique first", 
    alertErr: "There is an unexpected error",
    modalTitle: "AI Coach Analysis",
    issue: "Biomechanical Issue",
    plan: "Action Plan",
    power: "Power Generation",
    export: "Export Report",
    gotIt: "Got it, Coach!",
    unknown: "Unknown Technique",
    techs: {
      high_clear: "High Clear",
      smash: "Smash ➔",
      smash_standard: "Standard Smash",
      smash_stick: "Stick Smash",
      smash_jump: "Jump Smash",
      smash_slice: "Slice Smash",
      half_smash: "Half Smash",
      drop_shot: "Drop Shot ➔",
      drop_slice: "Slice Drop",
      drop_reverse_slice: "Reverse Slice Drop",
      net_shot: "Net Shot",
      net_spin: "Net Spin",
    }
  },
  zh: {
    title: "羽毛球 AI 教练",
    placeholder: "请选择技术动作",
    dragDrop: "拖拽视频到此处",
    clickBrowse: "或点击浏览文件",
    dropUpload: "松开鼠标完成上传！",
    btnUpload: "上传并分析",
    btnUploading: "上传中...",
    analyzing: "🏸 正在分析，请稍候...",
    errorReq: "❌ 网络请求错误，请重试",
    alertFile: "请先选择一个视频文件",
    alertAction: "请先选择一个技术动作", 
    alertErr: "发生未知错误",
    modalTitle: "AI 教练分析报告",
    issue: "动作诊断",
    plan: "改进方案",
    power: "发力技巧",
    export: "导出报告",
    gotIt: "明白，教练！",
    unknown: "未知动作",
    techs: {
      high_clear: "高远球",
      smash: "杀球 ➔",
      smash_standard: "重杀",
      smash_stick: "点杀",
      smash_jump: "跳杀",
      smash_slice: "劈杀",
      half_smash: "突击半场",
      drop_shot: "吊球 ➔",
      drop_slice: "劈吊",
      drop_reverse_slice: "滑板吊球",
      net_shot: "放网",
      net_spin: "搓球",
    }
  }
};

export default function Home() {
  const [file, setFile] = useState<File | null>(null);
  const [action, setAction] = useState<string>("");
  const [isLoading, setLoading] = useState<boolean>(false);
  const [hasError, setError] = useState<boolean>(false);
  const [feedback, setFeedback] = useState<CoachFeedback | null>(null);
  const [isDragging, setIsDragging] = useState<boolean>(false);
  const [showModal, setShowModal] = useState(false);

  // 2. 新增语言状态，默认英文
  const [lang, setLang] = useState<'en' | 'zh'>('en');
  const t = i18n[lang]; // 当前语言包

  const fileInputRef = useRef<HTMLInputElement>(null);

  // 动作列表：label 使用动态字典映射
  const techniques = [
    { id: 'high_clear', label: t.techs.high_clear },
    {
      id: 'smash',
      label: t.techs.smash,
      children: [
        { id: 'smash_standard', label: t.techs.smash_standard },
        { id: 'smash_stick', label: t.techs.smash_stick },
        { id: 'smash_jump', label: t.techs.smash_jump },
        { id: 'smash_slice', label: t.techs.smash_slice },
      ]
    },
    { id: 'half_smash', label: t.techs.half_smash },
    {
      id: 'drop_shot',
      label: t.techs.drop_shot,
      children: [
        { id: 'drop_slice', label: t.techs.drop_slice },
        { id: 'drop_reverse_slice', label: t.techs.drop_reverse_slice },
      ]
    },
    { id: 'net_shot', label: t.techs.net_shot },
    { id: 'net_spin', label: t.techs.net_spin },
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

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      setFile(e.target.files[0]);
    }
  }

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!action) {
      alert(t.alertAction);
      return;
    }
    if (!file) {
      alert(t.alertFile);
      return;
    }

    const formData = new FormData();
    formData.append("action", action);
    formData.append("video", file);
    formData.append("language", lang); // 核心：把语言传给后端，让 LLM 知道用什么语言回复

    setError(false);
    setLoading(true);
    setFeedback(null);

    try {
      const response = await fetch(
        'http://127.0.0.1:8001/upload-video',
        {
          method: 'POST',
          body: formData
        }
      );

      const result = await response.json();

      if (result.status === 'processed') {
        setFeedback(result.llm_feedback);
        setLoading(false);
        setFile(null);
        setShowModal(true);
      }
    } catch (error) {
      console.error(error);
      setError(true);
      setLoading(false);
      alert(t.alertErr);
    }
  }

  const handleExport = () => {
    if (!feedback) return;

    const techniqueName = techniques.flatMap(item => item.children ? item.children : item).find(item => item.id === action)?.label || t.unknown;
    const date = new Date().toLocaleDateString();

    const content = `🏸 ${t.modalTitle} 🏸\n\n` +
      `Date: ${date}\n` +
      `Technique Analyzed: ${techniqueName}\n` +
      `----------------------------------------\n\n` +
      `[${t.issue}]\n${feedback.problem}\n\n` +
      `[${t.plan}]\n${feedback.improvement}\n\n` +
      `[${t.power}]\n${feedback.power_technique}\n`;

    const blob = new Blob([content], { type: 'text/plain;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `Badminton_Analysis_${Date.now()}.txt`;

    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
  };

  return (
    <main className='relative min-h-screen p-8 flex flex-col items-center justify-center bg-gray-50'>

      {/* 右上角语言切换器 */}
      <div className="absolute top-6 right-6 flex bg-white rounded-lg shadow-sm border border-gray-200 p-1">
        <button
          onClick={() => setLang('en')}
          className={`px-3 py-1.5 rounded-md text-sm font-medium transition-colors ${lang === 'en' ? 'bg-blue-50 text-blue-600' : 'text-gray-500 hover:text-gray-700'}`}
        >
          EN
        </button>
        <button
          onClick={() => setLang('zh')}
          className={`px-3 py-1.5 rounded-md text-sm font-medium transition-colors ${lang === 'zh' ? 'bg-blue-50 text-blue-600' : 'text-gray-500 hover:text-gray-700'}`}
        >
          中文
        </button>
      </div>

      <h1 className='text-3xl font-bold mb-8 text-center text-gray-800'>{t.title}</h1>

      <form onSubmit={handleSubmit} className='flex flex-col gap-6 w-full max-w-md bg-white p-6 rounded-xl shadow-sm border border-gray-100'>

        <div className="relative group">
          <div className='border border-gray-300 rounded-lg p-3 bg-white text-gray-700 cursor-default flex justify-between items-center group-hover:ring-2 group-hover:ring-blue-500 transition-all'>
            <span>{techniques.flatMap(item => item.children ? item.children : item).find(item => item.id === action)?.label || t.placeholder}</span>
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

        <div
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onDrop={handleDrop}
          onClick={() => fileInputRef.current?.click()}
          className={`
            border-2 border-dashed rounded-xl p-10 text-center transition-all duration-200 cursor-pointer
            ${isDragging
              ? 'border-red-500 bg-red-50 scale-[1.02]'
              : 'border-gray-300 bg-gray-50 hover:border-blue-400'}
          `}
        >
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
              <p className="text-lg">{isDragging ? t.dropUpload : t.dragDrop}</p>
              <p className="text-sm mt-1 opacity-70">{t.clickBrowse}</p>
            </div>
          )}
        </div>

        <button
          type='submit'
          disabled={isLoading}
          className={`text-white p-3 rounded-lg font-bold transition-all shadow-md active:scale-[0.98] ${isLoading ? 'bg-blue-400 cursor-not-allowed' : 'bg-blue-600 hover:bg-blue-700'
            }`}
        >
          {isLoading ? t.btnUploading : t.btnUpload}
        </button>

        {isLoading && (
          <div className="flex flex-col items-center gap-3 py-2">
            <span className="text-gray-600 font-medium text-sm">{t.analyzing}</span>
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
            {t.errorReq}
          </div>
        )}
      </form>

      {showModal && feedback && (
        <div className="fixed inset-0 z-[100] flex items-center justify-center p-4 sm:p-6">
          <div
            className="absolute inset-0 bg-black/60 backdrop-blur-sm"
            onClick={() => setShowModal(false)}
          />

          <div className="relative bg-white w-full max-w-2xl max-h-[90vh] overflow-y-auto rounded-2xl shadow-2xl flex flex-col animate-in fade-in zoom-in duration-300">
            <div className="sticky top-0 bg-white border-b border-gray-100 px-6 py-4 flex justify-between items-center">
              <h2 className="text-xl font-bold text-gray-800 flex items-center gap-2">
                <span className="text-2xl">🏸</span> {t.modalTitle}
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
              <div className="p-6 space-y-8">
              {/* 1. Problem - 带有警告图标 */}
              <section>
                <div className="flex items-center gap-3 mb-3">
                  <div className="flex items-center justify-center w-8 h-8 rounded-full bg-red-100 text-red-500">
                    <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
                    </svg>
                  </div>
                  <h3 className="font-bold text-gray-800 uppercase tracking-wider text-sm">{t.issue}</h3>
                </div>
                <div className="bg-red-50 border border-red-100 rounded-xl p-5 text-gray-800 leading-relaxed shadow-sm">
                  {feedback.problem}
                </div>
              </section>

              {/* 2. Improvement - 带有任务清单图标 */}
              <section>
                <div className="flex items-center gap-3 mb-3">
                  <div className="flex items-center justify-center w-8 h-8 rounded-full bg-blue-100 text-blue-500">
                    <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2m-6 9l2 2 4-4" />
                    </svg>
                  </div>
                  <h3 className="font-bold text-gray-800 uppercase tracking-wider text-sm">{t.plan}</h3>
                </div>
                <div className="bg-blue-50 border border-blue-100 rounded-xl p-5 text-gray-800 leading-relaxed shadow-sm">
                  {/* 兼容数组或字符串的渲染方式 */}
                  {Array.isArray(feedback.improvement) ? (
                    <ul className="space-y-3">
                      {feedback.improvement.map((step, index) => (
                        <li key={index} className="flex gap-3">
                          <span className="font-bold text-blue-600 select-none">{index + 1}.</span>
                          <span>{step}</span>
                        </li>
                      ))}
                    </ul>
                  ) : (
                    <p className="whitespace-pre-wrap">{feedback.improvement}</p>
                  )}
                </div>
              </section>

              {/* 3. Power Technique - 带有闪电图标 */}
              <section>
                <div className="flex items-center gap-3 mb-3">
                  <div className="flex items-center justify-center w-8 h-8 rounded-full bg-amber-100 text-amber-500">
                    <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M13 10V3L4 14h7v7l9-11h-7z" />
                    </svg>
                  </div>
                  <h3 className="font-bold text-gray-800 uppercase tracking-wider text-sm">{t.power}</h3>
                </div>
                <div className="bg-amber-50 border border-amber-100 rounded-xl p-5 text-gray-800 leading-relaxed shadow-sm">
                  {feedback.power_technique}
                </div>
              </section>
            </div>
            </div>

            <div className="p-6 border-t border-gray-50 flex justify-end gap-3 bg-gray-50/50">
              <button
                onClick={handleExport}
                className="bg-white border border-gray-300 text-gray-700 px-5 py-2 rounded-lg font-medium hover:bg-gray-50 transition-all active:scale-95 flex items-center gap-2 shadow-sm"
              >
                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
                </svg>
                {t.export}
              </button>
              <button
                onClick={() => setShowModal(false)}
                className="bg-gray-900 text-white px-6 py-2 rounded-lg font-medium hover:bg-gray-800 transition-all active:scale-95 shadow-sm"
              >
                {t.gotIt}
              </button>
            </div>
          </div>
        </div>
      )}
    </main>
  )
}