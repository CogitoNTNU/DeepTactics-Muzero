import { useState } from "react";
import { useScreenSize } from "../hooks/use-screen-size";
import { PixelTrail } from "../components/ui/pixel-trail";
import { GooeyFilter } from "../components/ui/gooey-filter";
import { Hero45 } from "../components/ui/hero45";
import { ArrowUpRight } from "lucide-react";

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

  const handleMuzeroPaperClick = () => {
    window.open('https://www.deepmind.com/blog/muzero-mastering-go-chess-shogi-and-atari-without-rules', '_blank');
  };

  return (
    <div className="flex flex-col items-center justify-center pt-40 pb-20 space-y-12 text-white">
      <div className="text-center max-w-4xl mx-auto space-y-6">
        <h1 className="text-4xl font-bold">Welcome to MuZero Implementation</h1>
        <div className="space-y-4 text-lg">
          <p>
            This project is our implementation of MuZero, the groundbreaking reinforcement learning algorithm developed by DeepMind. 
            We've built this from scratch following the official research paper, aiming to recreate its remarkable ability to master 
            games without prior knowledge of their rules.
          </p>
          <p>
            Our implementation focuses on three classic games: Othello, TicTacToe, and CartPole, demonstrating MuZero's 
            versatility across different types of environments. We've carefully followed the architecture and training 
            procedures outlined in the original paper while optimizing for performance and clarity.
          </p>
        </div>
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

        <button
          onClick={handleMuzeroPaperClick}
          className="text-white text-7xl z-10 font-bold group flex items-center gap-4 hover:text-blue-400 transition-colors duration-300"
          aria-label="Read the MuZero paper by DeepMind"
        >
          Muzero-Paper
          <ArrowUpRight className="w-8 h-8 transform group-hover:translate-x-1 group-hover:-translate-y-1 transition-transform duration-300" />
        </button>
      </div>

      {/* Features Section */}
      <Hero45 
        badge="MuZero Features"
        heading="Advanced Neural Network Architecture"
        imageSrc="/images/neural-network.svg"
        imageAlt="Neural Network Architecture"
      >
        <div className="space-y-6 text-white">
          <p className="text-lg">
            Our implementation features a sophisticated neural network architecture with three key components:
          </p>
          <ul className="list-disc pl-6 space-y-3">
            <li>
              <strong>Representation Network:</strong> Processes the game state into a hidden representation, capturing essential features.
            </li>
            <li>
              <strong>Dynamics Network:</strong> Predicts the next state and rewards, enabling internal planning without explicit rules.
            </li>
            <li>
              <strong>Prediction Network:</strong> Evaluates positions and suggests optimal actions through policy and value predictions.
            </li>
          </ul>
          <p className="text-lg mt-4">
            The backend is built with FastAPI, providing a robust and efficient infrastructure for model training and inference.
            Our implementation ensures high performance through parallel processing and optimized state management.
          </p>
        </div>
      </Hero45>

      {/* Test Backend Section */}
      <div className="text-center space-y-4 max-w-2xl mx-auto text-white">
        <div className="space-y-4">
          <p className="text-lg">
            Want to see it in action? Test our backend connection below to get started with the MuZero implementation.
          </p>
          <button
            onClick={testBackend}
            className="px-6 py-3 bg-blue-500 text-white rounded-lg hover:bg-blue-600 transition-colors duration-300"
          >
            Test Backend Connection
          </button>
          {response && (
            <p className="mt-4 text-lg font-medium bg-white/10 backdrop-blur-sm px-4 py-2 rounded-lg">
              {response}
            </p>
          )}
        </div>
      </div>
    </div>
  );
}
  