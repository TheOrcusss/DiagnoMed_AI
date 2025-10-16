import React from "react";

const recentPatients = [
  { name: "Aarav Sharma", age: "45y", conditions: "Hypertension", status: "Diagnosed", statusColor: "text-green-600" },
  { name: "Saanvi Patel", age: "29y", conditions: "Diabetes", status: "In Progress", statusColor: "text-blue-600" },
  { name: "Rohan Singh", age: "60y", conditions: "Heart Disease", status: "Pending", statusColor: "text-yellow-600" },
];

export default function RecentPatients() {
  return (
    <section className="mt-8">
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        {recentPatients.map((p, idx) => (
          <div key={idx} className="bg-white dark:bg-panel-dark shadow rounded-xl p-6">
            <p className="font-medium text-gray-900 dark:text-text-light">{p.name}</p>
            <p className="text-sm text-gray-500 dark:text-text-blue">{p.age} · {p.conditions}</p>
            <span className={`${p.statusColor} text-sm font-semibold`}>{p.status}</span>
          </div>
        ))}
      </div>
    </section>
  );
}
