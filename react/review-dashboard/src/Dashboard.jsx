import React, { useEffect, useState } from "react";

function Dashboard() {
  const [data, setData] = useState(null);
  const [error, setError] = useState("");
  const [darkMode, setDarkMode] = useState(false);

  useEffect(() => {
    fetch("http://127.0.0.1:5000/get_latest_reviews")
      .then((res) => {
        if (!res.ok) throw new Error("Error fetching data");
        return res.json();
      })
      .then(setData)
      .catch((err) => {
        console.error(err);
        setError("Failed to fetch review data.");
      });
  }, []);

  const getGradeStyle = (grade) => {
    switch (grade) {
      case "A":
        return { icon: "🟢", color: "bg-emerald-500 text-white" };
      case "B":
        return { icon: "🟡", color: "bg-yellow-400 text-black" };
      case "C":
        return { icon: "🟠", color: "bg-orange-400 text-white" };
      default:
        return { icon: "🔴", color: "bg-red-500 text-white" };
    }
  };

  const toggleMode = () => setDarkMode((prev) => !prev);

  const gradeInfo = data ? getGradeStyle(data.grade) : {};

  return (
    <div
      className={`min-h-screen transition-all duration-300 ${
        darkMode
          ? "bg-gradient-to-br from-gray-900 via-gray-800 to-gray-900 text-white"
          : "bg-gradient-to-br from-blue-50 via-purple-50 to-pink-50 text-gray-900"
      }`}
    >
      <div className="max-w-4xl mx-auto px-4 py-6 font-sans">
        {/* Header */}
        <div className="flex justify-between items-center mb-8">
          <h1 className="text-4xl font-bold text-center w-full">
            Product Review Dashboard
          </h1>
          <button
            onClick={toggleMode}
            className="absolute right-6 top-6 bg-gray-300 hover:bg-gray-400 dark:bg-gray-700 dark:hover:bg-gray-600 px-3 py-1 rounded"
          >
            {darkMode ? "☀️" : "🌙"}
          </button>
        </div>

        {error && <div className="text-red-400 text-center">{error}</div>}
        {!data && !error && <div className="text-center">Loading...</div>}

        {data && (
          <>
            {/* Product Info */}
            <div
              className={`flex items-center gap-5 p-6 rounded-2xl shadow-md mb-8 ${
                darkMode ? "bg-gray-800" : "bg-white"
              }`}
            >
              <img
                src={data.productImage}
                alt="Product"
                className="w-28 h-28 object-contain rounded-xl border"
              />
              <div>
                <h2 className="text-2xl font-semibold">{data.productName}</h2>
                <div
                  className={`inline-flex items-center gap-2 mt-2 px-4 py-1 rounded-full text-lg font-medium shadow-sm ${gradeInfo.color}`}
                >
                  <span>{gradeInfo.icon}</span>
                  Grade: {data.grade}
                </div>
              </div>
            </div>

            {/* Top Reviews */}
            <div
              className={`p-6 rounded-2xl shadow-md ${
                darkMode ? "bg-gray-800" : "bg-white"
              }`}
            >
              <h3 className="text-2xl font-bold mb-4">Top 5 Reviews</h3>
              {data.top_reviews.map((review, i) => (
                <div key={i} className="mb-5 last:mb-0">
                  <div
                    className={`relative bg-gradient-to-br from-indigo-100 to-blue-100 dark:from-gray-700 dark:to-gray-800 p-4 rounded-lg shadow-md`}
                  >
                    <p className="italic">"{review.Review}"</p>
                    <div className="absolute bottom-[-10px] left-6 w-4 h-4 bg-inherit transform rotate-45" />
                  </div>
                </div>
              ))}
            </div>
          </>
        )}
      </div>
    </div>
  );
}

export default Dashboard;
