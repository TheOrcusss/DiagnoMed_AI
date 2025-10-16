import React, { useEffect, useState, useRef } from "react";
import { FileText, Eye, Loader2, X, AlertCircle } from "lucide-react";
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
        else setError("Failed to fetch cases.");
      } catch (err) {
        setError("Server error. Try again later.");
      } finally {
        setLoading(false);
      }
    };
    fetchCases();
  }, []);

  const downloadPDF = async () => {
    if (!reportRef.current) return;
    const input = reportRef.current;
    const canvas = await html2canvas(input, { scale: 2, useCORS: true });
    const imgData = canvas.toDataURL("image/png");
    const pdf = new jsPDF("p", "mm", "a4");
    const pdfWidth = pdf.internal.pageSize.getWidth();
    const pdfHeight = (canvas.height * pdfWidth) / canvas.width;
    pdf.addImage(imgData, "PNG", 0, 0, pdfWidth, pdfHeight);
    pdf.save(`${selectedCase.patient_name || "Patient_Report"}.pdf`);
  };

  return (
    <div className="min-w-full mx-auto p-6 bg-gray-50 min-h-screen">
      <h1 className="flex items-center gap-3 text-3xl font-bold text-blue-600 mb-6">
        <FileText className="w-7 h-7" />
        Patient Diagnoses
      </h1>

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
          <table className="min-w-full text-sm align-middle">
            <thead className="bg-blue-600 text-white text-left">
              <tr>
                <th className="py-3 px-4">Patient Name</th>
                <th className="py-3 px-4">Symptoms</th>
                <th className="py-3 px-4">Prediction</th>
                <th className="py-3 px-4">Confidence</th>
                <th className="py-3 px-4">Actions</th>
              </tr>
            </thead>
            <tbody>
              {cases.map((c) => (
                <tr
                  key={c.id}
                  className="text-black hover:bg-blue-50 border-b transition"
                >
                  <td className="py-3 px-4">{c.patient_name}</td>
                  <td className="py-3 px-4">{c.symptoms}</td>
                  <td className="py-3 px-4 text-red-600">{c.cnn_output}</td>
                  <td className="py-3 px-4">{c.analysis_output}</td>
                  <td className="py-3 px-4 text-center">
                    <button
                      onClick={() => setSelectedCase(c)}
                      className="!bg-blue-600 hover:bg-blue-700 text-white px-4 py-2 rounded-lg"
                    >
                      <Eye className="w-4 h-4 inline mr-1" /> View
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {selectedCase && (
        <div className="fixed inset-0 bg-white flex justify-center items-center z-50">
          <div
            ref={reportRef}
            className="bg-white border border-gray-300 rounded-2xl shadow-2xl p-8 w-[90%] md:w-[75%] max-h-[90vh] overflow-y-auto relative"
          >
            <button
              onClick={() => setSelectedCase(null)}
              className="absolute top-4 right-4 text-gray-500 hover:text-black"
            >
              <X className="w-5 h-5" />
            </button>

            <h2 className="text-2xl font-bold text-blue-700 mb-4">
              Patient Report
            </h2>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
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
                  <strong>Prediction:</strong> {selectedCase.cnn_output}
                </p>
                <p>
                  <strong>Confidence:</strong> {selectedCase.analysis_output}
                </p>
              </div>

              <div className="flex flex-col items-center">
                {selectedCase.gradcam_url ? (
                  <div className="p-3 border rounded-xl shadow-md bg-gray-50">
                    <p className="text-center text-gray-700 mb-2">
                      🔥 Grad-CAM Heatmap
                    </p>
                    <img
                      src={selectedCase.gradcam_url}
                      alt="GradCAM Heatmap"
                      className="rounded-lg object-contain max-h-[450px] w-full"
                    />
                  </div>
                ) : (
                  <p className="text-gray-600 text-center">
                    ⚠️ No Grad-CAM image available.
                  </p>
                )}
              </div>
            </div>

            <div className="flex justify-end mt-6">
              <button
                onClick={downloadPDF}
                className="!bg-green-600 hover:bg-green-700 text-white px-5 py-2 rounded-xl"
              >
                <FileText className="w-4 h-4 inline mr-2" /> Download Report
                (PDF)
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
