import React, { useState } from "react";

const patientsData = [
  { id: 1, name: "Aarav Sharma", age: 45, condition: "Hypertension", lastVisit: "2025-08-12" },
  { id: 2, name: "Saanvi Patel", age: 29, condition: "Diabetes", lastVisit: "2025-07-30" },
  { id: 3, name: "Rohan Singh", age: 60, condition: "Heart Disease", lastVisit: "2025-07-25" },
  { id: 4, name: "Ananya Gupta", age: 37, condition: "Asthma", lastVisit: "2025-08-01" },
  { id: 5, name: "Kabir Mehta", age: 52, condition: "Hypertension", lastVisit: "2025-08-05" },
  { id: 6, name: "Isha Verma", age: 30, condition: "Allergy", lastVisit: "2025-08-03" },
  { id: 7, name: "Vivaan Reddy", age: 40, condition: "Migraine", lastVisit: "2025-08-06" },
  { id: 8, name: "Diya Nair", age: 25, condition: "Thyroid", lastVisit: "2025-08-02" },
  { id: 9, name: "Arjun Kapoor", age: 35, condition: "Asthma", lastVisit: "2025-08-07" },
  { id: 10, name: "Meera Joshi", age: 38, condition: "Diabetes", lastVisit: "2025-08-04" },
];

export default function Patients() {
  const [search, setSearch] = useState("");

  const filtered = patientsData.filter((p) =>
    p.name.toLowerCase().includes(search.toLowerCase())
  );

  return (
    <div className="p-6 min-w-screen bg-gray-50 dark:bg-gradient-to-br dark:from-background-gradient1 dark:via-background-gradient2 dark:to-background-gradient3">
      <h2 className="text-2xl font-bold text-gray-800 dark:text-text-light">Patients</h2>

      <input
        type="text"
        placeholder="Search patients..."
        value={search}
        onChange={(e) => setSearch(e.target.value)}
        className="mt-4 mb-6 border px-3 py-2 rounded-lg w-1/3 bg-white dark:bg-panel-dark text-gray-900 dark:text-text-light border-gray-300 dark:border-gray-600"
      />

      <table className="w-full bg-white dark:bg-panel-dark shadow rounded-lg overflow-hidden">
        <thead className="bg-gray-100 dark:bg-panel-deep text-left">
          <tr>
            <th className="p-3 text-gray-700 dark:text-text-light">Name</th>
            <th className="p-3 text-gray-700 dark:text-text-light">Age</th>
            <th className="p-3 text-gray-700 dark:text-text-light">Condition</th>
            <th className="p-3 text-gray-700 dark:text-text-light">Last Visit</th>
          </tr>
        </thead>
        <tbody>
          {filtered.map((p) => (
            <tr
              key={p.id}
              className="border-t hover:bg-gray-50 dark:hover:bg-panel-deep"
            >
              <td className="p-3 text-gray-900 dark:text-text-light">{p.name}</td>
              <td className="p-3 text-gray-900 dark:text-text-light">{p.age}</td>
              <td className="p-3 text-gray-900 dark:text-text-light">{p.condition}</td>
              <td className="p-3 text-gray-900 dark:text-text-light">{p.lastVisit}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}