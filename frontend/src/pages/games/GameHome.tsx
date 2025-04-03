import { Link, Outlet, useLocation } from 'react-router-dom';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '../../components/ui/card';
import { games } from '../../config/gameData';

export default function GameHome() {
  const location = useLocation();

  // Check if we are already on a specific game page
  const isGamePage = location.pathname.includes('/watch/');

  return (
    <div className="pt-40 pb-20 px-6 text-white">
      <div className="max-w-7xl mx-auto">
        <h2 className="text-3xl font-bold mb-6">MuZero Games</h2>

        {/* Only show the links if we are not on a game page */}
        {!isGamePage && (
          <div className="flex space-x-6">
            {games.map((game) => (
              <Link key={game.title} to={game.link} className="px-4 py-2 rounded-lg">
                <Card className="w-80 border border-white/10 bg-white/5 backdrop-blur-sm shadow-lg rounded-xl overflow-hidden">
                  <CardHeader className="flex items-center flex-col text-center">
                    <img
                      src={game.imageUrl}
                      alt={game.title}
                      className="w-40 h-40 rounded-full border-2 border-white/20 object-cover"
                    />
                    <CardTitle className="mt-4 text-2xl font-semibold text-white">{game.title}</CardTitle>
                    <CardDescription className="text-lg text-gray-300 font-medium">{game.description}</CardDescription>
                  </CardHeader>
                  <CardContent className="flex justify-center gap-4 py-4"></CardContent>
                </Card>
              </Link>
            ))}
          </div>
        )}

        {/* Render the Outlet where the game page components will be shown */}
        <div className="mt-12">
          <Outlet />
        </div>
      </div>
    </div>
  );
}
