import React, { useState } from "react";
import { History, Activity } from "lucide-react";

export default function PatientHistoryPage() {
  // Demo data
  const [history] = useState([
    { id: 1, condition: "Flu", treatment: "Paracetamol", date: "2024-08-20" },
    { id: 2, condition: "Hypertension", treatment: "Amlodipine", date: "2024-05-12" },
  ]);

  const [ongoing] = useState([
    { id: 1, condition: "Chest Pain", status: "Under Analysis" },
  ]);

  return (
    <div className="max-w-5xl mx-auto p-6 space-y-8">
      {/* Treatment History */}
      <div className="bg-white shadow rounded-xl p-6">
        <h1 className="flex items-center gap-2 text-2xl font-bold !text-blue-600 mb-4">
          <History className="w-6 h-6" />
          Treatment History
        </h1>
        <ul className="space-y-3">
          {history.map((item) => (
            <li
              key={item.id}
              className="p-4 border border-black rounded-lg bg-gray-50 hover:bg-gray-100 transition"
            >
              <p>
                <span className="font-medium !text-black">Condition:</span>{" "}
                <span className="!text-gray-600">{item.condition}</span>
              </p>
              <p>
                <span className="font-medium !text-black">Treatment:</span>{" "}
                <span className="!text-gray-600">{item.treatment}</span>
              </p>
              <p className="text-sm !text-gray-500">{item.date}</p>
            </li>
          ))}
        </ul>
      </div>

      {/* Ongoing Diagnoses */}
      <div className="bg-white shadow rounded-xl p-6">
        <h1 className="flex items-center gap-2 text-2xl font-bold !text-blue-600 mb-4">
          <Activity className="w-6 h-6" />
          Ongoing Diagnosis
        </h1>
        <ul className="space-y-3">
          {ongoing.map((item) => (
            <li
              key={item.id}
              className="p-4 border border-black rounded-lg bg-gray-50 hover:bg-gray-100 transition"
            >
              <p>
                <span className="font-medium !text-black">Condition:</span>{" "}
                <span className="!text-gray-600">{item.condition}</span>
              </p>
              <p>
                <span className="font-medium !text-black">Status:</span>{" "}
                <span className="!text-gray-600">{item.status}</span>
              </p>
            </li>
          ))}
        </ul>
      </div>
    </div>
  );
}
