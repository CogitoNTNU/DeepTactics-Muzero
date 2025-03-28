import { Routes, Route } from 'react-router-dom';
import Navbar from './components/Navbar';
import Home from './pages/Home';
import MuZeroConfig from './pages/MuZeroConfig';
import About from './pages/About';
import GameHome from './pages/games/GameHome';
import Othello from './pages/games/Othello';
import CartPole from './pages/games/CartPole';

function App() {
  return (
    <div className="min-h-screen bg-[#242424] text-[rgba(255,255,255,0.87)]">
      <Navbar />
      <div className="container mx-auto p-4">
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/config" element={<MuZeroConfig />} />
          <Route path="/about" element={<About />} />
          {/* Nested routes for games */}
          <Route path="/watch" element={<GameHome />}>
            <Route path="othello" element={<Othello />} />
            <Route path="cartpole" element={<CartPole />} />
          </Route>
        </Routes>
      </div>
    </div>
  );
}

export default App;
