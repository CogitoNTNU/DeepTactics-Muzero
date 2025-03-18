import { useState } from "react";

export default function Home() {
  const [response, setResponse] = useState("");
  
  const testBackend = async () => {
    try {
      const res = await fetch("/api/ping");  // Internal Docker network request
      const data = await res.json();
      setResponse(data.message);
    } catch (error) {
      setResponse("Error connecting to backend");
      console.error("API call failed:", error);
    }
  }; 

  return (
    <div className="flex flex-col items-center justify-center py-20">
      <h1 className="text-4xl font-bold">Welcome to MuZero</h1>

      <p className="mt-4 text-lg">This is the main page.</p>

      {/* API Test Button */}
      <button
        onClick={testBackend}
        className="px-4 py-2 bg-blue-500 rounded hover:bg-teal-600"
      >
        Test Backend
      </button>

      {/* Show API Response */}
      <p className="mt-4">{response}</p>
    </div>
  );
}
  