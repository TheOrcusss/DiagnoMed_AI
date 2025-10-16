import React, { useEffect, useState, useRef } from "react";
import { FileText, Eye, Loader2, X, Download, AlertCircle } from "lucide-react";
import jsPDF from "jspdf";
import html2canvas from "html2canvas";

const BACKEND_URL = import.meta.env.VITE_API_URL;

export default function NewDiagnosis() {
  const [cases, setCases] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [selectedCase, setSelectedCase] = useState(null);
  const reportRef = useRef();

  useEffect(() => {
    const fetchCases = async () => {
      try {
        const response = await fetch(`${BACKEND_URL}/api/doctor/cases`);
        const data = await response.json();
        if (response.ok) setCases(data);
        else setError("Failed to fetch cases from backend.");
      } catch (err) {
        console.error("❌ Error fetching cases:", err);
        setError("Server error. Please try again later.");
      } finally {
        setLoading(false);
      }
    };
    fetchCases();
  }, []);

  // 🧾 Convert report to PDF
  const downloadPDF = async () => {
    if (!reportRef.current) return;
    const input = reportRef.current;

    const canvas = await html2canvas(input, {
      scale: 2,
      useCORS: true,
    });

    const imgData = canvas.toDataURL("image/png");
    const pdf = new jsPDF("p", "mm", "a4");
    const pdfWidth = pdf.internal.pageSize.getWidth();
    const pdfHeight = (canvas.height * pdfWidth) / canvas.width;
    pdf.addImage(imgData, "PNG", 0, 0, pdfWidth, pdfHeight);
    pdf.save(`${selectedCase.patient_name || "Patient_Report"}.pdf`);
  };

  return (
    <div className="min-w-screen mx-auto p-6 bg-gray-50 min-h-screen">
      <h1 className="flex items-center gap-3 text-3xl font-bold text-blue-600 mb-6">
        <FileText className="w-7 h-7" />
        Patient Diagnoses
      </h1>

      {/* --- Loading / Error / Table --- */}
      {loading ? (
        <div className="flex justify-center items-center py-20">
          <Loader2 className="w-8 h-8 animate-spin text-blue-600" />
        </div>
      ) : error ? (
        <div className="flex flex-col items-center py-20 text-red-600">
          <AlertCircle className="w-8 h-8 mb-2" />
          <p>{error}</p>
        </div>
      ) : cases.length === 0 ? (
        <p className="text-center text-gray-600 mt-10">
          No patient submissions yet.
        </p>
      ) : (
        <div className="overflow-x-auto bg-white rounded-2xl shadow-md border border-gray-200">
          <table className="min-w-full text-sm">
            <thead className="bg-blue-600 text-white text-left">
              <tr>
                <th className="py-3 px-4 font-semibold">Patient Name</th>
                <th className="py-3 px-4 font-semibold">Symptoms</th>
                <th className="py-3 px-4 font-semibold">CNN Output</th>
                <th className="py-3 px-4 font-semibold">Bayesian Output</th>
                <th className="py-3 px-4 text-center font-semibold">Actions</th>
              </tr>
            </thead>
            <tbody>
              {cases.map((c) => (
                <tr
                  key={c.id}
                  className="hover:bg-blue-50 border-b border-gray-100 transition"
                >
                  <td className="py-3 px-4 text-black">
                    {c.patient_name || "Unknown"}
                  </td>
                  <td className="py-3 px-4 text-gray-700 truncate max-w-xs">
                    {c.symptoms || "—"}
                  </td>
                  <td className="py-3 px-4 font-medium text-blue-600">
                    {c.cnn_output || "Pending"}
                  </td>
                  <td className="py-3 px-4 text-gray-700">
                    {c.analysis_output || "Pending"}
                  </td>
                  <td className="py-3 px-4 text-center">
                    <button
                      onClick={() => setSelectedCase(c)}
                      className="flex items-center justify-center gap-2 !bg-blue-600 hover:bg-blue-700 text-white px-4 py-2 rounded-lg font-medium transition"
                    >
                      <Eye className="w-4 h-4" /> View Report
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {/* --- Modal: Patient Report --- */}
      {selectedCase && (
        <div className="fixed inset-0 bg-white flex justify-center items-center z-50">
          <div className="bg-white border border-gray-300 rounded-2xl shadow-2xl p-8 w-[90%] md:w-[75%] max-h-[90vh] overflow-y-auto relative transition-all duration-300 hover:shadow-[0_10px_40px_rgba(0,0,0,0.2)]">
            <button
              onClick={() => setSelectedCase(null)}
              className="absolute top-4 right-4 text-gray-500 hover:text-black transition"
            >
              <X className="w-5 h-5" />
            </button>

            <h2 className="text-2xl font-bold text-blue-700 mb-5 border-b-2 border-blue-600 pb-2">
              Patient Report
            </h2>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-6 items-start">
              {/* LEFT SIDE: DETAILS */}
              <div className="space-y-2 text-gray-800">
                <p>
                  <strong>Name:</strong> {selectedCase.patient_name}
                </p>
                <p>
                  <strong>Age:</strong> {selectedCase.age || "N/A"}
                </p>
                <p>
                  <strong>Blood Type:</strong>{" "}
                  {selectedCase.blood_type || "N/A"}
                </p>
                <p>
                  <strong>Symptoms:</strong> {selectedCase.symptoms}
                </p>
                <p>
                  <strong>CNN Output:</strong>{" "}
                  <span className="text-blue-700 font-semibold">
                    {selectedCase.cnn_output}
                  </span>
                </p>
                <p>
                  <strong>Bayesian Output:</strong>{" "}
                  <span className="text-gray-700 font-semibold">
                    {selectedCase.analysis_output}
                  </span>
                </p>
              </div>

              {/* RIGHT SIDE: IMAGES */}
              <div className="flex flex-col items-center gap-6">
                <div className="p-3 rounded-xl border border-gray-300 bg-gray-50 shadow-lg w-full">
                  <p className="font-medium text-gray-700 mb-2 text-center">
                    🩻 Patient Uploaded X-ray
                  </p>
                  <img
                    src={selectedCase.image_url}
                    alt="Patient X-ray"
                    className="rounded-lg border shadow-md object-contain max-h-[400px] w-full hover:scale-[1.02] transition-transform"
                  />
                </div>

                {selectedCase.gradcam_url && (
                  <div className="p-3 rounded-xl border border-gray-300 bg-gray-50 shadow-lg w-full">
                    <p className="font-medium text-gray-700 mb-2 text-center">
                      🔥 GradCAM Heatmap
                    </p>
                    <img
                      src={selectedCase.gradcam_url}
                      alt="GradCAM Heatmap"
                      className="rounded-lg border shadow-md object-contain max-h-[400px] w-full hover:scale-[1.02] transition-transform"
                    />
                  </div>
                )}
              </div>
            </div>

            <div className="flex justify-end mt-8">
              <button
                onClick={() => window.print()}
                className="!bg-green-600 hover:bg-green-700 text-white px-5 py-2 rounded-xl font-medium flex items-center gap-2 transition"
              >
                <FileText className="w-4 h-4" /> Download Report (PDF)
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
