import { useState } from "react";
import { useScreenSize } from "../hooks/use-screen-size";
import { PixelTrail } from "../components/ui/pixel-trail";
import { GooeyFilter } from "../components/ui/gooey-filter";
import { ArrowUpRight, Brain, Cpu, Zap } from "lucide-react";

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
    <div className="flex flex-col items-center justify-center pt-40 pb-20 text-white">
      {/* Main Content Container - 2/3 width */}
      <div className="w-full max-w-[66%] mx-auto space-y-16">
        {/* Introduction Section */}
        <div className="text-center space-y-6">
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
        <div className="w-full flex justify-center">
          <div className="relative w-[85vw] min-h-[500px] flex flex-col items-center justify-center overflow-hidden rounded-2xl">
            <img
              src="https://images.aiscribbles.com/34fe5695dbc942628e3cad9744e8ae13.png?v=60d084"
              alt="impressionist painting"
              className="w-full h-full object-cover absolute inset-0 opacity-70"
            />
            <div className="absolute inset-0 bg-black/20" />
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
        </div>

        {/* Neural Network Architecture Section */}
        <div className="space-y-8 text-center">
          <h2 className="text-3xl font-semibold">Neural Network Architecture</h2>
          <p className="text-lg">
            Our implementation features a sophisticated neural network architecture that forms the core of the MuZero algorithm.
            This architecture consists of three key components working together to achieve state-of-the-art performance:
          </p>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mt-8">
            <div className="flex flex-col items-center space-y-4 p-6 border border-white/10 rounded-lg bg-white/5">
              <div className="p-3 rounded-full bg-blue-500/20">
                <Brain className="w-6 h-6 text-blue-400" />
              </div>
              <h3 className="font-semibold">Representation Network</h3>
              <p className="text-sm text-gray-300">
                Processes the game state into a hidden representation, capturing essential features for decision making.
              </p>
            </div>
            <div className="flex flex-col items-center space-y-4 p-6 border border-white/10 rounded-lg bg-white/5">
              <div className="p-3 rounded-full bg-purple-500/20">
                <Cpu className="w-6 h-6 text-purple-400" />
              </div>
              <h3 className="font-semibold">Dynamics Network</h3>
              <p className="text-sm text-gray-300">
                Predicts the next state and rewards, enabling internal planning without explicit game rules.
              </p>
            </div>
            <div className="flex flex-col items-center space-y-4 p-6 border border-white/10 rounded-lg bg-white/5">
              <div className="p-3 rounded-full bg-teal-500/20">
                <Zap className="w-6 h-6 text-teal-400" />
              </div>
              <h3 className="font-semibold">Prediction Network</h3>
              <p className="text-sm text-gray-300">
                Evaluates positions and suggests optimal actions through policy and value predictions.
              </p>
            </div>
          </div>
          <img 
            src="/public/nn.png" 
            alt="Neural Network Architecture" 
            className="w-full max-w-3xl mx-auto mt-8 rounded-lg border border-white/10"
          />
        </div>

        {/* Test Backend Section */}
        <div className="text-center space-y-4">
          <div className="border border-white/10 bg-white/5 backdrop-blur-sm p-8 rounded-lg space-y-6">
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
    </div>
  );
}
  