import { useState } from "react";
import { useScreenSize } from "../hooks/use-screen-size";
import { PixelTrail } from "../components/ui/pixel-trail";
import { GooeyFilter } from "../components/ui/gooey-filter";

export default function Home() {
  const [response, setResponse] = useState("");
  const screenSize = useScreenSize();
  
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
    <div className="flex flex-col items-center justify-center py-20 space-y-12">
      <div>
        <h1 className="text-4xl font-bold">Welcome to MuZero</h1>
        <p className="mt-4 text-lg">This is the main page.</p>

        {/* API Test Button */}
        <button
          onClick={testBackend}
          className="mt-4 px-4 py-2 bg-blue-500 rounded hover:bg-teal-600"
        >
          Test Backend
        </button>

        {/* Show API Response */}
        <p className="mt-4">{response}</p>
      </div>

      {/* Gooey Effect Section */}
      <div className="relative w-full h-[400px] flex flex-col items-center justify-center">
        <img
          src="https://images.unsplash.com/photo-1635070041078-e363dbe005cb?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=2000&q=80"
          alt="AI background"
          className="w-full h-full object-cover absolute inset-0 opacity-70"
        />

        <GooeyFilter id="gooey-filter-pixel-trail" strength={5} />

        <div
          className="absolute inset-0 z-0"
          style={{ filter: "url(#gooey-filter-pixel-trail)" }}
        >
          <PixelTrail
            pixelSize={screenSize.lessThan("md") ? 24 : 32}
            fadeDuration={0}
            delay={500}
            pixelClassName="bg-white"
          />
        </div>

        <h2 className="text-white text-6xl z-10 font-bold">
          Muzero-Paper
        </h2>
      </div>
    </div>
  );
}
  