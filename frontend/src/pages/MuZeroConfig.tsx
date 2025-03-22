import InputList from "../components/InputList"
import { environmentParameters } from "../config/muzeroParameters";


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
