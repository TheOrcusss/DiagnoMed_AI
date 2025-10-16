import React, { useState } from "react";
import { FileText } from "lucide-react";

export default function SymptomsPage() {
  const [symptoms, setSymptoms] = useState("");

  const quickSymptoms = [
    "Headache",
    "Fever",
    "Cough",
    "Fatigue",
    "Nausea",
    "Sore throat",
    "Shortness of breath",
  ];

    const handleSubmit = (e) => {
    e.preventDefault();
    if (!symptoms.trim()) return;

    // Save to localStorage so UploadScansPage can read it
    localStorage.setItem("symptoms", symptoms);
    alert("✅ Symptoms saved! Now please upload your scan.");
    setSymptoms("");
  };


  const handleToggleSymptom = (symptom) => {
    const symptomList = symptoms.split(",").map((s) => s.trim()).filter(Boolean);

    if (symptomList.includes(symptom)) {
      // Remove symptom if already in list
      const updated = symptomList.filter((s) => s !== symptom).join(", ");
      setSymptoms(updated);
    } else {
      // Add symptom
      setSymptoms(symptomList.length ? `${symptoms}, ${symptom}` : symptom);
    }
  };

  const isSymptomSelected = (symptom) => {
    const symptomList = symptoms.split(",").map((s) => s.trim());
    return symptomList.includes(symptom);
  };

  return (
    <div className="max-w-5xl mx-auto p-6">
      {/* Header */}
      <h1 className="flex items-center gap-3 text-3xl font-bold text-blue-600 mb-6">
        <FileText className="w-7 h-7" />
        Enter Your Symptoms
      </h1>

      {/* Quick Symptom Suggestions */}
      <div className="mb-4">
        <p className="text-gray-600 mb-2 font-medium">Quick Symptoms:</p>
        <div className="flex flex-wrap gap-2">
          {quickSymptoms.map((symptom) => (
            <button
              key={symptom}
              type="button"
              onClick={() => handleToggleSymptom(symptom)}
              className={`px-3 py-1 rounded-full text-sm transition ${
                isSymptomSelected(symptom)
                  ? "!bg-blue-600 text-white"
                  : "!bg-blue-100 text-blue-700 hover:bg-blue-200"
              }`}
            >
              {symptom}
            </button>
          ))}
        </div>
      </div>

      {/* Form */}
      <form
        onSubmit={handleSubmit}
        className="bg-white shadow-md rounded-2xl p-6 space-y-4"
      >
        <label className="block !text-gray-700 font-medium mb-2">
          Describe your symptoms:
        </label>
        <textarea
          value={symptoms}
          onChange={(e) => setSymptoms(e.target.value)}
          placeholder="Describe your symptoms in detail, e.g., headache, mild fever, fatigue, or nausea..."
          rows={6}
          required
          className="w-full p-4 border border-gray-300 rounded-xl resize-none focus:outline-none focus:ring-2 focus:ring-blue-400 transition placeholder-gray-400 !text-black"
        />

        <button
          type="submit"
          className="w-full py-3 !bg-blue-600 text-white rounded-xl font-semibold text-lg hover:bg-blue-700 transition"
        >
          Submit
        </button>
      </form>
    </div>
  );
}
