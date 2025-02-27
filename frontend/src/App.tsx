import { useState } from "react";
import reactLogo from "./assets/react.svg";
import viteLogo from "../public/vite.svg";
import Button from "./components/Button";

function App() {
  const [count, setCount] = useState(0);

  return (
    <div className="flex flex-col items-center justify-center min-h-screen bg-[#242424] text-[rgba(255,255,255,0.87)]">
      {/* Logos */}
      <div className="flex justify-center space-x-8">
        <a href="https://vite.dev" target="_blank">
          <img src={viteLogo} className="h-24 transition filter hover:drop-shadow-[0_0_8px_#646cffaa] animate-logo-spin" alt="Vite logo" />
        </a>
        <a href="https://react.dev" target="_blank">
          <img src={reactLogo} className="h-24 transition filter hover:drop-shadow-[0_0_8px_#61dafbaa] animate-logo-spin" alt="React logo" />
        </a>
      </div>

      {/* Titles */}
      <h1 className="text-5xl font-extrabold mt-6">Vite + React</h1>
      <h2 className="text-2xl font-semibold mt-2">DeepTactics - MuZero</h2>
      <h3 className="text-xl font-bold text-blue-500 mt-2">@Cogito NTNU</h3>

      {/* Card & Button */}
      <div className="mt-6 p-6 bg-gray-800 rounded-lg shadow-md flex flex-col items-center">
        <button
          className="h-10 px-6 text-white font-medium bg-red-500 rounded-lg transition hover:bg-red-600 disabled:opacity-50"
          onClick={() => setCount((count) => count + 1)}
        >
          Count is {count}
        </button>
        <p className="mt-4 text-gray-400">
          Edit <code className="font-mono bg-gray-700 px-2 py-1 rounded">src/App.tsx</code> and save to test HMR.
        </p>
      </div>

      <div className="p-6 bg-gray-800 rounded-lg shadow-md">
        <h2 className="text-xl font-bold">Card Title</h2>
        <p className="text-gray-400">This is some card content.</p>
      </div>
      <div className="flex items-center justify-center bg-gray-900">
        <Button />
      </div>

      <div className="cursor-pointer bg-blue-500 text-white p-3 rounded" onClick={() => alert("Clicked!")}>
        Click me
      </div>



      {/* Footer */}
      <p className="mt-4 text-gray-500">Click on the Vite and React logos to learn more.</p>
    </div>
  );
}

export default App;
