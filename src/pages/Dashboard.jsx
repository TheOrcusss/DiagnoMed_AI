import React from "react";
import { Link } from "react-router-dom";
import PatientChart from "../components/PatientChart";
import DiagnosisDistribution from "../components/DiagnosisDistribution";
import RecentPatients from "../components/RecentPatients";
import Card from "../components/Card";

export default function Dashboard() {
  return (
    <main className="p-6 w-full space-y-4">
      {/* Header Section */}
      <header className="flex flex-col md:flex-row items-start md:items-center justify-between bg-white p-6 rounded-2xl shadow-sm border border-gray-200">
        {/* Left: Welcome message */}
        <div>
          <h2 className="text-2xl font-bold text-gray-900">
            Welcome Back, Dr. Yash
          </h2>
          <p className="mt-1 text-gray-600">
            Here’s what’s happening with your patients today.
          </p>
        </div>

        {/* Right: Navigation Button */}
        <div className="mt-4 md:mt-0">
          <Link
            to="/NewDiagnosis"
            className="inline-block px-5 py-3 rounded-lg !bg-blue-600 hover:!bg-blue-700 !text-white font-medium shadow transition"
          >
            View Patients
          </Link>
        </div>
      </header>

      {/* Stats Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-6 w-full">
        <Card title="Total Patients" value="1,248" color="text-gray-900" />
        <Card title="Pending Diagnoses" value="32" color="text-red-500" />
        <Card title="Completed Reports" value="846" color="text-green-600" />
        <Card title="Activity" value="+12%" color="text-blue-600" />
      </div>

      {/* Charts */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-4">
        <PatientChart />
        <DiagnosisDistribution />
      </div>

      {/* Recent Patients Section */}
      <section className="w-full">
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-xl font-semibold text-black">Recent Patients</h2>
          <Link
            to="/patients"
            className="text-blue-600 hover:underline font-medium"
          >
            View All →
          </Link>
        </div>
        <RecentPatients />
      </section>
    </main>
  );
}
