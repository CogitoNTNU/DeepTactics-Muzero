import { Link, Outlet } from 'react-router-dom';

export default function GameHome() {
  return (
    <div className="p-6">
      <h2 className="text-3xl font-bold mb-4">MuZero Games</h2>
      <div className="flex space-x-4">
        <Link to="./othello" className="text-blue-500 hover:underline">Othello</Link>
        <Link to="./cartpole" className="text-blue-500 hover:underline">CartPole</Link>
        <Link to="./tictactoe" className="text-blue-500 hover:underline">TicTacToe</Link>
      </div>

      <div className="mt-8">
        <Outlet />
      </div>
    </div>
  );
}
