// import { Link } from "react-router-dom";

// export default function GameHome() {
//   return (
//     <div className="space-y-4">
//       <h2 className="text-3xl font-bold">MuZero Games</h2>
//       <ul className="mt-4 list-disc list-inside">
//         <li><Link to="othello" className="text-blue-400">Watch Othello</Link></li>
//         <li><Link to="cartpole" className="text-blue-400">Watch CartPole</Link></li>
//       </ul>
//     </div>
//   );
// }

import { Link, Outlet } from 'react-router-dom';

export default function GameHome() {
  return (
    <div className="p-6">
      <h2 className="text-3xl font-bold mb-4">MuZero Games</h2>
      <div className="flex space-x-4">
        <Link to="othello" className="text-blue-500 hover:underline">Othello</Link>
        <Link to="cartpole" className="text-blue-500 hover:underline">CartPole</Link>
      </div>

      <div className="mt-8">
        <Outlet />
      </div>
    </div>
  );
}
