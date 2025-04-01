import { useState } from "react";
import { useScreenSize } from "../hooks/use-screen-size";
import { PixelTrail } from "../components/ui/pixel-trail";
import { GooeyFilter } from "../components/ui/gooey-filter";
import { Hero45 } from "../components/ui/hero45";

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
    <div className="flex flex-col items-center justify-center py-20 space-y-12 text-white">
      <div className="text-center">
        <h1 className="text-4xl font-bold">Welcome to MuZero</h1>
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
            fadeDuration={100}
            delay={16}
            pixelClassName="bg-white"
          />
        </div>

        <p className="text-white text-7xl z-10 font-bold">
          Muzero-Paper
        </p>
      </div>

      {/* Test Backend Section */}
      <div className="text-center space-y-4">
        <p className="text-lg">This is the main page.</p>
        <button
          onClick={testBackend}
          className="px-6 py-3 bg-blue-500 rounded-lg hover:bg-teal-600 transition-colors"
        >
          Test Backend
        </button>
        {response && (
          <p className="mt-4 text-lg">{response}</p>
        )}
      </div>

      {/* Features Section */}
      <Hero45 
        badge="MuZero Features"
        heading="State-of-the-Art Reinforcement Learning"
        imageSrc="https://images.unsplash.com/photo-1620712943543-bcc4688e7485?q=80&w=2940&auto=format&fit=crop"
        imageAlt="AI visualization"
      />
    </div>
  );
}
  