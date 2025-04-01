import { Link, Outlet } from 'react-router-dom';

export default function GameHome() {
  return (
    <div className="pt-40 pb-20 px-6 text-white">
      <div className="max-w-7xl mx-auto">
        <h2 className="text-3xl font-bold mb-6">MuZero Games</h2>
        <div className="flex space-x-6">
          <Link to="./othello" className="px-4 py-2 rounded-lg bg-white/10 hover:bg-white/20 transition-colors duration-300">Othello</Link>
          <Link to="./cartpole" className="px-4 py-2 rounded-lg bg-white/10 hover:bg-white/20 transition-colors duration-300">CartPole</Link>
          <Link to="./tictactoe" className="px-4 py-2 rounded-lg bg-white/10 hover:bg-white/20 transition-colors duration-300">TicTacToe</Link>
        </div>

        <div className="mt-12">
          <Outlet />
        </div>
      </div>
    </div>
  );
}
