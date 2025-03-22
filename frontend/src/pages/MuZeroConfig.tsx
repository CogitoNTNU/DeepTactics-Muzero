import InputList from "../components/InputList"
import { environmentParameters, selfPlaySettings, explorationSettings, trainingParameters, neuralNetworkSettings, puctParameters, loggingSettings } from "../config/muzeroParameters";


export default function MuZeroConfig() {
  return (
    <div className="flex flex-col items-center justify-center py-20">
      <h1 className="text-4xl font-bold">Welcome to the config site!</h1>
      <p className="mt-4 text-lg">Finetune the training configs here!</p>
      <div className="pt-10 flex flex-row justify-between w-full">
        <InputList title="Enviroment" parameters={environmentParameters} />
        <InputList title="Self Play" parameters={selfPlaySettings} />
        <InputList title="Exploration" parameters={explorationSettings} />
        <InputList title="Training" parameters={trainingParameters} />
        <InputList title="Neural Network" parameters={neuralNetworkSettings} />
        <InputList title="Puct" parameters={puctParameters} />
        <InputList title="Logging" parameters={loggingSettings} />
      </div>
    </div>
  );
}
