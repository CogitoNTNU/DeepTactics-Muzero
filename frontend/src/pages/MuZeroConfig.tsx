import InputWithTooltip from "../components/InputWithTooltip"

export default function MuZeroConfig() {
  return (
    <div className="flex flex-col items-center justify-center py-20">
      <h1 className="text-4xl font-bold">Welcome to the config site!</h1>
      <p className="mt-4 text-lg">Finetune the training configs here!</p>
      <div className="pt-10">
        <InputWithTooltip labelText="Max_moves" defaultValue={50_000} tooltipMessage="Enter the maximum number of moves." />
      </div>
    </div>
  );
}
