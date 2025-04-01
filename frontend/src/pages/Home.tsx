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
      <div className="relative w-full h-full min-h-[600px] flex flex-col items-center justify-center gap-8 bg-black text-center text-pretty">
        <img
          src="https://images.aiscribbles.com/34fe5695dbc942628e3cad9744e8ae13.png?v=60d084"
          alt="impressionist painting"
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

        <p className="text-white text-7xl z-10 font-bold w-1/2">
          Muzero-Paper
        </p>
      </div>
    </div>
  );
}
  