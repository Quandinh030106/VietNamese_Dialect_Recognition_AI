import { useState, useRef } from 'react';
import axios from 'axios';
import { Bar } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  BarElement,
  Title,
  Tooltip,
  Legend,
} from 'chart.js';
import { Mic, Upload, Play, Square, Loader2, Music, CheckCircle } from 'lucide-react';

// Đăng ký ChartJS
ChartJS.register(CategoryScale, LinearScale, BarElement, Title, Tooltip, Legend);

function App() {
  const [activeTab, setActiveTab] = useState('upload'); // 'upload' hoặc 'record'
  const [file, setFile] = useState(null);
  const [audioUrl, setAudioUrl] = useState(null);
  const [isRecording, setIsRecording] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  const mediaRecorderRef = useRef(null);
  const chunksRef = useRef([]);

  // --- XỬ LÝ UPLOAD FILE ---
  const handleFileChange = (event) => {
    const selectedFile = event.target.files[0];
    if (selectedFile) {
      setFile(selectedFile);
      setAudioUrl(URL.createObjectURL(selectedFile));
      setResult(null);
      setError(null);
    }
  };

  // --- XỬ LÝ GHI ÂM ---
  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      mediaRecorderRef.current = new MediaRecorder(stream);
      chunksRef.current = [];

      mediaRecorderRef.current.ondataavailable = (e) => {
        if (e.data.size > 0) chunksRef.current.push(e.data);
      };

      mediaRecorderRef.current.onstop = () => {
        const blob = new Blob(chunksRef.current, { type: 'audio/wav' });
        const recordedFile = new File([blob], "recording.wav", { type: "audio/wav" });
        setFile(recordedFile);
        setAudioUrl(URL.createObjectURL(blob));
      };

      mediaRecorderRef.current.start();
      setIsRecording(true);
      setResult(null);
      setError(null);
    } catch (err) {
      console.error("Error accessing microphone:", err);
      setError("Không thể truy cập microphone. Vui lòng kiểm tra quyền trên trình duyệt.");
    }
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current && isRecording) {
      mediaRecorderRef.current.stop();
      setIsRecording(false);
      mediaRecorderRef.current.stream.getTracks().forEach(track => track.stop());
    }
  };

  // --- GỌI API BACKEND ---
  const handlePredict = async () => {
    if (!file) return;

    setIsLoading(true);
    setError(null);

    const formData = new FormData();
    formData.append("file", file);

    try {
      // GỌI TỚI BACKEND ĐANG CHẠY Ở PORT 8000
      const response = await axios.post("https://quan030106-vietnamese-dialect-api.hf.space/predict", formData, {
        headers: { "Content-Type": "multipart/form-data" },
      });
      setResult(response.data);
    } catch (err) {
      console.error(err);
      setError("Lỗi kết nối! Hãy chắc chắn Backend đang chạy (uvicorn main:app).");
    } finally {
      setIsLoading(false);
    }
  };

  // --- CẤU HÌNH BIỂU ĐỒ ---
  const chartData = result ? {
    labels: Object.keys(result.probabilities),
    datasets: [
      {
        label: 'Độ tin cậy (%)',
        data: Object.values(result.probabilities).map(val => (val * 100).toFixed(1)),
        backgroundColor: [
          'rgba(239, 68, 68, 0.6)',   // Red
          'rgba(249, 115, 22, 0.6)',  // Orange
          'rgba(234, 179, 8, 0.6)',   // Yellow
          'rgba(34, 197, 94, 0.6)',   // Green
          'rgba(59, 130, 246, 0.6)',  // Blue
          'rgba(168, 85, 247, 0.6)',  // Purple
        ],
        borderWidth: 1,
        borderRadius: 6,
      },
    ],
  } : null;

  const chartOptions = {
    responsive: true,
    plugins: {
      legend: { display: false },
      title: { display: true, text: 'Xác suất dự đoán', font: { size: 16 } },
    },
    scales: {
        y: { beginAtZero: true, max: 100 }
    }
  };

  return (
    <div className="min-h-screen flex items-center justify-center p-4 font-sans text-gray-800 bg-gradient-to-br from-blue-50 to-indigo-50">
      <div className="bg-white rounded-3xl shadow-2xl w-full max-w-5xl overflow-hidden flex flex-col md:flex-row min-h-[600px]">
        
        {/* CỘT TRÁI: INPUT */}
        <div className="w-full md:w-1/2 p-8 md:p-12 flex flex-col justify-between border-r border-gray-100 bg-white">
          <div>
            <h1 className="text-3xl font-extrabold text-indigo-900 mb-2 tracking-tight">Nhận diện Vùng miền</h1>
            <p className="text-gray-500 mb-8">AI phân tích giọng nói tiếng Việt</p>

            {/* TAB CHUYỂN ĐỔI */}
            <div className="flex bg-gray-100 p-1.5 rounded-xl mb-6">
              <button
                onClick={() => setActiveTab('upload')}
                className={`flex-1 py-2.5 rounded-lg text-sm font-semibold transition-all duration-200 flex items-center justify-center gap-2 ${
                  activeTab === 'upload' ? 'bg-white shadow-sm text-indigo-600' : 'text-gray-500 hover:text-gray-700'
                }`}
              >
                <Upload size={18} /> Tải file lên
              </button>
              <button
                onClick={() => setActiveTab('record')}
                className={`flex-1 py-2.5 rounded-lg text-sm font-semibold transition-all duration-200 flex items-center justify-center gap-2 ${
                  activeTab === 'record' ? 'bg-white shadow-sm text-red-500' : 'text-gray-500 hover:text-gray-700'
                }`}
              >
                <Mic size={18} /> Ghi âm
              </button>
            </div>

            {/* KHUNG UPLOAD */}
            {activeTab === 'upload' && (
              <div className="border-2 border-dashed border-indigo-100 rounded-2xl p-10 text-center hover:border-indigo-400 hover:bg-indigo-50 transition-all group">
                <input type="file" accept="audio/*" onChange={handleFileChange} className="hidden" id="file-upload" />
                <label htmlFor="file-upload" className="cursor-pointer flex flex-col items-center">
                  <div className="w-16 h-16 bg-indigo-100 rounded-full flex items-center justify-center text-indigo-500 mb-4 group-hover:scale-110 transition-transform">
                    <Music size={32} />
                  </div>
                  <span className="text-gray-700 font-medium group-hover:text-indigo-600">Chọn file âm thanh</span>
                  <span className="text-xs text-gray-400 mt-1">.wav, .mp3, .m4a</span>
                </label>
              </div>
            )}

            {/* KHUNG GHI ÂM */}
            {activeTab === 'record' && (
              <div className="flex flex-col items-center justify-center py-10 bg-gray-50 rounded-2xl border border-gray-100">
                {isRecording ? (
                  <div className="flex flex-col items-center">
                     <div className="relative">
                        <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-red-400 opacity-75"></span>
                        <div className="w-20 h-20 bg-red-500 rounded-full flex items-center justify-center relative z-10 text-white shadow-lg shadow-red-200">
                            <Mic size={40} />
                        </div>
                     </div>
                     <p className="text-red-500 font-medium mt-6 mb-4 animate-pulse">Đang thu âm...</p>
                     <button onClick={stopRecording} className="bg-white border border-red-200 text-red-500 hover:bg-red-50 px-6 py-2 rounded-full flex items-center gap-2 font-medium transition-all shadow-sm">
                       <Square size={16} fill="currentColor" /> Dừng lại
                     </button>
                  </div>
                ) : (
                  <button onClick={startRecording} className="group flex flex-col items-center">
                    <div className="w-20 h-20 bg-white border-4 border-gray-100 rounded-full flex items-center justify-center text-gray-400 shadow-sm group-hover:border-red-100 group-hover:text-red-500 transition-all duration-300">
                        <Mic size={40} />
                    </div>
                    <span className="mt-4 text-sm font-medium text-gray-500 group-hover:text-red-500">Nhấn để bắt đầu</span>
                  </button>
                )}
              </div>
            )}

            {/* AUDIO PLAYER */}
            {audioUrl && (
              <div className="mt-6 bg-indigo-50/50 p-4 rounded-xl border border-indigo-100 flex items-center gap-4">
                <div className="w-10 h-10 bg-indigo-600 rounded-full flex items-center justify-center text-white shrink-0">
                    <Play size={20} fill="currentColor" className="ml-0.5" />
                </div>
                <div className="flex-1 min-w-0">
                    <p className="text-sm font-medium text-indigo-900 truncate">{file?.name || "Bản ghi âm mới"}</p>
                    <audio controls src={audioUrl} className="w-full h-8 mt-1 opacity-80" />
                </div>
              </div>
            )}
          </div>

          {/* NÚT SUBMIT */}
          <div className="mt-8">
            {error && <div className="mb-4 p-3 bg-red-50 text-red-600 text-sm rounded-lg text-center border border-red-100">{error}</div>}
            
            <button
                onClick={handlePredict}
                disabled={!file || isLoading}
                className={`w-full py-4 rounded-xl font-bold text-white shadow-lg flex items-center justify-center gap-2 transition-all transform hover:-translate-y-0.5 active:translate-y-0 ${
                !file || isLoading 
                    ? 'bg-gray-300 cursor-not-allowed shadow-none' 
                    : 'bg-indigo-600 hover:bg-indigo-700 hover:shadow-indigo-200'
                }`}
            >
                {isLoading ? <><Loader2 className="animate-spin" /> Đang phân tích...</> : "Dự đoán ngay"}
            </button>
          </div>
        </div>

        {/* CỘT PHẢI: KẾT QUẢ */}
        <div className="w-full md:w-1/2 bg-gray-50 p-8 md:p-12 flex flex-col justify-center relative overflow-hidden">
            {/* Background trang trí */}
            <div className="absolute top-0 right-0 w-64 h-64 bg-indigo-200 rounded-full mix-blend-multiply filter blur-3xl opacity-20 -translate-y-1/2 translate-x-1/2"></div>
            <div className="absolute bottom-0 left-0 w-64 h-64 bg-pink-200 rounded-full mix-blend-multiply filter blur-3xl opacity-20 translate-y-1/2 -translate-x-1/2"></div>

          {result ? (
            <div className="relative z-10 animate-fade-in">
              <div className="text-center mb-10">
                <div className="inline-flex items-center gap-2 px-4 py-1.5 bg-green-100 text-green-700 rounded-full text-sm font-bold mb-4">
                    <CheckCircle size={16} /> Hoàn tất phân tích
                </div>
                <h2 className="text-5xl font-black text-indigo-900 mb-2 leading-tight">{result.prediction.split('(')[0]}</h2>
                <p className="text-indigo-400 font-medium">{result.prediction.split('(')[1]?.replace(')', '')}</p>
              </div>

              <div className="bg-white p-6 rounded-2xl shadow-sm border border-gray-100">
                <Bar options={chartOptions} data={chartData} />
              </div>
            </div>
          ) : (
            <div className="relative z-10 h-full flex flex-col items-center justify-center text-center text-gray-400">
                <div className="w-32 h-32 bg-gray-200/50 rounded-full flex items-center justify-center mb-6">
                    <Music size={48} className="text-gray-300" />
                </div>
                <h3 className="text-xl font-semibold text-gray-500 mb-2">Chưa có kết quả</h3>
                <p className="max-w-xs mx-auto">Vui lòng tải lên file hoặc ghi âm giọng nói để AI bắt đầu phân tích.</p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

export default App;