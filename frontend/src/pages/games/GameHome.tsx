import { Link, Outlet, useLocation } from 'react-router-dom';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '../../components/ui/card';

const games = [
  {
    title: 'Othello',
    description: 'Othello er et strategisk brettspill for to spillere der målet er å omringe og snu motstanderens brikker for å dominere brettet.',
    imageUrl: 'https://www.eothello.com/images/how_to_play_othello_0.png',
    link: './othello',
  },
  {
    title: 'Cartpole',
    description: 'CartPole er en klassisk kontrollproblem der målet er å balansere en stang på en vogn ved å justere vogna\'s bevegelser.',
    imageUrl: 'https://miro.medium.com/v2/resize:fit:1188/1*LVoKXR7aX7Y8npUSsYZdKg.png',
    link: './cartpole',
  },
  {
    title: 'TicTacToe',
    description: 'Tic-Tac-Toe er et spill der to spillere bytter på å plassere X eller O på et 3x3 rutenett for å få tre på rad.',
    imageUrl: 'https://cdn-icons-png.flaticon.com/512/10199/10199746.png',
    link: './tictactoe',
  },
];

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
