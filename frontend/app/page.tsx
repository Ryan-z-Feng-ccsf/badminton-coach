'use client'

import React, {useState } from 'react';
interface CoachFeedback{
  problem:string;
  improvement: string;
  power_technique:string;
}
export default function Home() {

  const [file, setFile] = useState<File | null>(null);
  const [action, setAction] = useState<string>("");
  const [isLoading, setLoading] = useState<boolean>(false);
  const [hasError, setError] = useState<boolean>(false);
  const [feedback, setFeedback] = useState<CoachFeedback|null>(null);


  const handleDragOver = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
  }
  const handleDragLeave = (e: React.DragEvent<HTMLElement>) => {
    e.preventDefault();
  }
  const handleDrop = (e: React.DragEvent<HTMLElement>) => {
    e.preventDefault();
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      const droppedFile = e.dataTransfer.files[0];
      setFile(droppedFile);
    }
  }
  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!file) {
      alert("Please select a video file first")
      return;
    }

    const formData = new FormData();
    formData.append("action", action);
    formData.append("video", file);

    if (hasError) {
      setError(false);
    }
    setLoading(true);

    try {
      const response = await fetch(
        'http://127.0.0.1:8000/upload-video',
        {
          method: 'POST',
          body: formData
        }
      );

      const result = await response.json();
      console.log("Result from the backend", result);
      alert("Request sent");


      if (result.status == 'processed') {
        const feedback = result.llm_feedback
        setLoading(false);
        setFeedback(feedback)
      }

    }
    catch (error) {
      console.log("Network request failed", error);
      setError(true);
      setLoading(false);
      alert('There is an unexpected error');
    }
  }
  return (
    <main className='p-8'>
      <h1 className='text-2xl font-bold mb-6'>Badminton AI - Video</h1>
      <form onSubmit={handleSubmit} className='flex flex-col gap-4 max-w-sm'>
        
        <select
          value={action}
          onChange={(e) => setAction(e.target.value)}
          className='border border-gray-300 rounded p-2 bg-white text-gray-700 focus:outline-none focus:ring-2 focus:ring-blue-500'
          required
        >
          <option value='' disabled>Select a technique</option>
          <option value='high_clear'>High Clear</option>
          <optgroup label='smash'>
            <option value='smash_standard'>Standard Smash</option>
            <option value='smash_stick'>Stick Smash</option>
            <option value='smash_jump'>Jump Smash</option>
            <option value='smash_slice'>Slice Smash</option>
          </optgroup>
          <option value='half_smash'>Half Smash</option>
          <optgroup label='drop_shot'>
            <option value='drop_slice'>Slice Drop</option>
            <option value='drop_reverse_slice'>Reverse Slice Drop</option>
          </optgroup>

          <option value='net_shot'>Net Shot</option>
          <option value='net_spin'>Net Spin</option>

        </select>
        <div
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onDrop={handleDrop}
          className='border-2 border-dashed border-gray-300 bg-gray-50 rounded-lg p-12 text-center'
        >

          {file ? (
            <p className="text-green-600 font-semibold">{file.name}</p>
          ) : (
            <p className="text-gray-500">Drag & drop your video here</p>
          )

          }

        </div>
        <button type='submit' className='bg-blue-600 text-white p-2 rounded'>
          Upload
        </button>
        {isLoading && (
          <div>
            🏸 Analyzing,wait a second
            <svg 
            viewBox="0 0 100 100" 
            className="w-16 h-16 text-blue-500"
          >
            {/* 核心动画逻辑：利用内联 CSS 让两组羽毛交替闪烁，模拟极速自转 */}
            <style>
              {`
                @keyframes spin-feather {
                  0%, 100% { opacity: 1; }
                  50% { opacity: 0.1; }
                }
                .feather-a { animation: spin-feather 0.15s infinite; }
                .feather-b { animation: spin-feather 0.15s infinite 0.075s; }
              `}
            </style>
            
            {/* 球托 (保持稳定) */}
            <path d="M40,75 C40,90 60,90 60,75 C60,70 40,70 40,75 Z" fill="currentColor" />
            
            {/* 羽毛 A 组 */}
            <path className="feather-a" d="M42,70 L20,15 L35,15 Z" fill="currentColor" opacity="0.8" />
            <path className="feather-a" d="M52,70 L60,10 L70,10 Z" fill="currentColor" opacity="0.8" />
            
            {/* 羽毛 B 组 (交替闪烁) */}
            <path className="feather-b" d="M48,70 L40,10 L50,10 Z" fill="currentColor" opacity="0.6" />
            <path className="feather-b" d="M58,70 L80,15 L65,15 Z" fill="currentColor" opacity="0.6" />
          </svg>

          </div>
        )}
        {hasError && (
          <div className='mt-8 p-4 bg-red-50 border border-red-200 rounded-lg text-red-600'>
            ❌ Network Request Error, please try again~

          </div>
        )}
        {feedback && (
          <div className='mt-8 p-4 bg-green-50 border border-green-200 rounded-lg text-gray-800 whitespace-pre-wrap'>
            <h3 className='font-bold text-green-700 mb-2'>AI Coach Feedbcak</h3>
            <p>{feedback.problem}</p>
            <p>{feedback.improvement}</p>
            <p>{feedback.power_technique}</p>

          </div>
        )}
      </form>

    </main>
  )

}