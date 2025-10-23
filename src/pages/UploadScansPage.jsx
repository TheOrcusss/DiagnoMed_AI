import React, { useState } from "react";
import { Upload } from "lucide-react";

const BACKEND_URL = import.meta.env.VITE_API_URL;

export default function UploadScansPage() {
  const [file, setFile] = useState(null);
  const [status, setStatus] = useState("");
  const [loading, setLoading] = useState(false);

  const handleFileChange = (e) => {
    setFile(e.target.files[0]);
  };

  const handleUpload = async (e) => {
    e.preventDefault();
    const symptoms = localStorage.getItem("symptoms");

    if (!file) return setStatus("❌ Please select a file first.");
    if (!symptoms) return setStatus("⚠️ Please fill the Symptoms page first.");

    const formData = new FormData();
    formData.append("name", "Anonymous Patient");
    formData.append("symptoms", symptoms);
    formData.append("image", file);

    try {
      setLoading(true);
      setStatus("⏳ Uploading scan...");

      const res = await fetch(`${BACKEND_URL}/api/patient/submit`, {
        method: "POST",
        body: formData,
      });

      const data = await res.json();
      if (res.ok) {
        setStatus("✅ Scan uploaded successfully!");
        localStorage.removeItem("symptoms");
        setFile(null);
      } else {
        setStatus("❌ Error: " + (data.error || "Something went wrong"));
      }
    } catch (err) {
      console.error(err);
      setStatus("❌ Server error, please try again later.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="max-w-5xl mx-auto p-6">
      {/* Header */}
      <h1 className="flex items-center gap-3 text-3xl font-bold !text-blue-600 mb-6">
        <Upload className="w-7 h-7" />
        Upload CT / MRI / X-ray Scans
      </h1>

      {/* Description */}
      <p className="mb-6 text-gray-600">
        Upload your scans and symptoms for doctor review. Our AI tool will
        analyze them for further diagnosis.
      </p>

      {/* Upload Form */}
      <form
        onSubmit={handleUpload}
        className="bg-white shadow-md rounded-2xl p-6 space-y-4"
      >
        <label className="flex flex-col items-center justify-center w-full p-6 border-2 border-dashed border-gray-300 rounded-2xl cursor-pointer hover:!border-blue-400 hover:!bg-blue-50 transition">
          <span className="!text-gray-500 mb-2">Click to select a file</span>
          <span className="text-sm !text-gray-400">
            Supported: PNG, JPG, PDF
          </span>
          <input
            type="file"
            accept="image/*,.pdf"
            onChange={handleFileChange}
            className="hidden"
          />
        </label>

        {file && (
          <p className="!text-gray-700 text-sm mt-2">
            Selected file: {file.name}
          </p>
        )}

        <button
          type="submit"
          disabled={loading}
          className={`w-full py-3 rounded-xl font-semibold text-lg transition ${
            loading
              ? "!bg-gray-400 cursor-not-allowed"
              : "!bg-blue-600 !text-white hover:!bg-blue-700"
          }`}
        >
          {loading ? "Uploading..." : "Upload"}
        </button>
      </form>

      {status && <p className="mt-4 !text-blue-700">{status}</p>}
    </div>
  );
}
