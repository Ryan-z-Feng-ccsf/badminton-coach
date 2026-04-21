'use client'

import React, { useState } from 'react';
export default function Home() {

  const [file, setFile] = useState<File | null>(null);
  const [action, setAction] = useState<string>("");


  const handleDragOver=(e: React.DragEvent<HTMLDivElement>)=>{
    e.preventDefault();
  }
  const handleDragLeave=(e:React.DragEvent<HTMLElement>)=>{
    e.preventDefault();
  }
  const handleDrop=(e:React.DragEvent<HTMLElement>)=>{
    e.preventDefault();
    if(e.dataTransfer.files&&e.dataTransfer.files.length>0){
      const droppedFile= e.dataTransfer.files[0];
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
    try {
      const response = await fetch(
        'http://127.0.0.1:8000/upload-video',
        {
          method: 'POST',
          body: formData
        }
      );
      
      const result = await response.json();
      console.log("Result from the backend",result);
      alert("Request sent");
    }
    catch(error) {
      console.log("Network request failed",error);

    }
  }
  return (
    <main className='p-8'>
      <h1 className='text-2xl font-bold mb-6'>Badminton AI - Video</h1>
      <form onSubmit={handleSubmit} className='flex flex-col gap-4 max-w-sm'>
        <div 
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        className='border-2 border-dashed border-gray-300 bg-gray-50 rounded-lg p-12 text-center'
        >

          {file ?(
            <p className="text-green-600 font-semibold">{file.name}</p>
          ):(
            <p className="text-gray-500">Drag & drop your video here</p>
          )
            
          }
          
        </div>
        <button type='submit' className='bg-blue-600 text-white p-2 rounded'>
          Upload
        </button>
      </form>

    </main>
  )

}