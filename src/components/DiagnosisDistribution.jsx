import React from "react";
import { PieChart, Pie, Cell, Tooltip, Legend, ResponsiveContainer } from "recharts";

const data = [
  { name: "Cardiology", value: 400 },
  { name: "Neurology", value: 300 },
  { name: "Orthopedics", value: 250 },
  { name: "Oncology", value: 200 },
  { name: "Other", value: 150 },
];

const COLORS = ["#3B82F6", "#10B981", "#F59E0B", "#EF4444", "#6366F1"];

export default function DiagnosisDistribution() {
  return (
    <div className="bg-white shadow rounded-xl p-6 w-full">
      <h2 className="text-lg font-bold text-gray-700 mb-4">Diagnosis Distribution</h2>
      <div className="h-72">
        <ResponsiveContainer width="100%" height="100%">
          <PieChart>
            <Pie
              data={data}
              cx="50%"
              cy="50%"
              labelLine={false}
              label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}
              outerRadius={100}
              fill="#8884d8"
              dataKey="value"
            >
              {data.map((entry, index) => (
                <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
              ))}
            </Pie>
            <Tooltip />
            <Legend />
          </PieChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}