"use client";
import { useState } from "react";
import { predictVideo } from "@/lib/api";

export default function Home() {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [isDragging, setIsDragging] = useState(false);
  const [error, setError] = useState<string | null>(null);

  type PredictionResult = {
    prediction: string;
    confidence: number;
  };
  const [result, setResult] = useState<PredictionResult | null>(null); // Store API response

  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) validateFile(file);
  };

  const validateFile = (file: File) => {
    const allowedTypes = ["video/mp4", "video/quicktime"];
    const maxSize = 50 * 1024 * 1024; // 50MB

    if (!allowedTypes.includes(file.type)) {
      setError("Only MP4, MOV videos are allowed.");
      setSelectedFile(null);
    } else if (file.size > maxSize) {
      setError("File size should be less than 50MB.");
      setSelectedFile(null);
    } else {
      setError(null);
      setSelectedFile(file);
      console.log(selectedFile?.type)
    }
  };

  const handleAnalyze = async () => {
    if (!selectedFile) return;
    
    setIsLoading(true);
    setError(null);
    setResult(null);
    
    try {
      const data = await predictVideo(selectedFile);
      setResult({
        prediction: data.prediction,
        confidence: data.confidence,
      });
    } catch (err: unknown) {
      if (err instanceof Error) {
        setError(err.message);
      } else {
        setError("Error analyzing file. Please try again.");
      }
    } finally {
      setIsLoading(false);
    }
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);

    const file = e.dataTransfer.files?.[0];
    if (file) validateFile(file);
  };

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = () => {
    setIsDragging(false);
  };

  return (
    <div className="min-h-screen bg-[#0b0f14] text-white flex items-center justify-center px-6">

      <div className="w-full max-w-3xl">

        {/* HEADER BLOCK */}
        <div className="mb-10">
          <h1 className="text-4xl font-semibold tracking-tight">
            DeepDetect
          </h1>
          <p className="text-gray-400 text-sm mt-2">
            Video integrity analysis system for synthetic media detection
          </p>
        </div>

        {/* MAIN PANEL */}
        <div className="bg-[#111827] border border-[#1f2937] rounded-xl p-6 shadow-[0_0_0_1px_rgba(255,255,255,0.02)]">

          {/* DROPZONE */}
          <div
            onDrop={handleDrop}
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
            className={`relative border-2 border-dashed rounded-lg p-10 transition
              ${
                isDragging
                  ? "border-blue-400 bg-[#0f172a]"
                  : "border-[#2b3442] hover:border-[#3b82f6]"
              }
            `}
          >
            <input
              type="file"
              accept="video/mp4,video/quicktime"
              className="absolute inset-0 opacity-0 cursor-pointer"
              onChange={handleFileChange}
              disabled={isLoading}
            />

            {!selectedFile ? (
              <div className="text-center">
                <p className="text-white font-medium">
                  Drop video for analysis
                </p>
                <p className="text-xs text-gray-500 mt-2">
                  MP4 / MOV • max 50MB
                </p>
              </div>
            ) : (
              <p className="text-center text-blue-300 font-medium">
                {selectedFile.name}
              </p>
            )}
          </div>

          {/* ERROR */}
          {error && (
            <div className="mt-4 text-sm text-red-400">
              {error}
            </div>
          )}

          {/* PREVIEW + ACTION */}
          {selectedFile && (
            <div className="mt-6 grid grid-cols-1 md:grid-cols-2 gap-6">

              {/* VIDEO */}
              <div className="border border-[#1f2937] rounded-lg overflow-hidden">
                <video
                  src={URL.createObjectURL(selectedFile)}
                  controls
                  className="w-full"
                />
              </div>

              {/* CONTROL PANEL */}
              <div className="flex flex-col justify-between">

                <div className="space-y-2 text-sm text-gray-400">
                  <p className="text-white font-medium">
                    File metadata
                  </p>
                  <p>{selectedFile.name}</p>
                  <p>
                    {(selectedFile.size / 1024 / 1024).toFixed(2)} MB
                  </p>
                </div>

                <button
                  onClick={handleAnalyze}
                  disabled={isLoading}
                  className={`mt-6 w-full py-3 rounded-lg font-medium transition
                    ${
                      isLoading
                        ? "bg-[#1f2937] text-gray-500"
                        : "bg-blue-600 hover:bg-blue-500 text-white"
                    }
                  `}
                >
                  {isLoading ? "Analyzing..." : "Run Detection"}
                </button>

              </div>
            </div>
          )}

          {/* RESULT */}
          {result && (
            <div className="mt-8 border-t border-[#1f2937] pt-6">

              <div className="flex items-center justify-between">
                <span className="text-gray-400 text-sm">
                  Prediction
                </span>

                <span
                  className={`text-xl font-semibold ${
                    result.prediction === "Real"
                      ? "text-green-400"
                      : "text-red-400"
                  }`}
                >
                  {result.prediction}
                </span>
              </div>

              <div className="mt-4">
                <div className="flex justify-between text-xs text-gray-500 mb-2">
                  <span>Confidence</span>
                  <span>{(result.confidence * 100).toFixed(1)}%</span>
                </div>

                <div className="h-2 bg-[#1f2937] rounded-full overflow-hidden">
                  <div
                    className={`h-full ${
                      result.prediction === "Real"
                        ? "bg-green-500"
                        : "bg-red-500"
                    }`}
                    style={{
                      width: `${result.confidence * 100}%`,
                    }}
                  />
                </div>
              </div>

            </div>
          )}

        </div>
      </div>
    </div>
  );
}
