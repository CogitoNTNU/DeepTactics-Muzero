import { Link } from 'react-router-dom';

export default function Navbar() {
  return (
    <nav className="bg-gray-800 text-gray-300 p-4 shadow-md">
      <div className="max-w-7xl mx-auto flex justify-between items-center">
        <img src="/cogito_white.svg" alt="Cogito" className="h-8 w-8 mr-2" />
        <div className="text-xl font-bold w-1/2">
          DeepTactics
        </div>
        <div className="flex space-x-8">
          <Link to="/" className="hover:text-blue-400 transition">DeepTactics</Link>
          <Link to="/config" className="hover:text-blue-400 transition">Config</Link>
          <Link to="/watch" className="hover:text-blue-400 transition">Watch MuZero</Link>
          <Link to="/about" className="hover:text-blue-400 transition">About us</Link>
        </div>
      </div>
    </nav>
  );
};
