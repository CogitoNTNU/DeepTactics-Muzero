import InputList from "../components/InputList"
const environmentParameters = [
  { labelText: "Action Space Size", defaultValue: 10, tooltipMessage: "The number of possible actions in the environment." },
  { labelText: "Input Planes", defaultValue: 5, tooltipMessage: "The number of input planes representing the state." },
  { labelText: "Height", defaultValue: 8, tooltipMessage: "The height of the board/grid." },
  { labelText: "Width", defaultValue: 8, tooltipMessage: "The width of the board/grid." },
  { labelText: "Num Input Moves", defaultValue: 4, tooltipMessage: "The number of past moves used as input." },
  { labelText: "Max Moves", defaultValue: 100.0, tooltipMessage: "The maximum number of moves in a game." }
];


export default function MuZeroConfig() {
  return (
    <div className="flex flex-col items-center justify-center py-20">
      <h1 className="text-4xl font-bold">Welcome to the config site!</h1>
      <p className="mt-4 text-lg">Finetune the training configs here!</p>
      <div className="pt-10">
        <InputList parameters={environmentParameters} />
      </div>
    </div>
  );
}
