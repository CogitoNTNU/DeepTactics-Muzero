import { CodeBlock } from "../components/CodeBlock";
import InputList from "../components/InputList"
import { configValues } from "../config/muzeroParameters";

export default function MuZeroConfig() {
  return (
    <div className="flex flex-col items-center justify-center py-20">
      <h1 className="text-4xl font-bold">Welcome to the config site!</h1>
      <p className="mt-4 text-lg">Finetune the training configs here!</p>
      <div className="pt-10 flex flex-row justify-between w-full">
        <InputList title="Enviroment" parameters={configValues.environment} />
        <InputList title="Self Play" parameters={configValues.selfPlay} />
        <InputList title="Exploration" parameters={configValues.exploration} />
        <InputList title="Training" parameters={configValues.training} />
        <InputList title="Neural Network" parameters={configValues.neuralNetwork} />
        <InputList title="Puct" parameters={configValues.puct} />
        <InputList title="Logging" parameters={configValues.logging} />
      </div>
      <div className="p-10">
        <CodeBlock
          code={`from typing import Optional
from src.utils.minmaxstats import KnownBounds

class Config:
    def __init__(
        self,
        render=True,
        known_bounds: Optional[KnownBounds] = None,
        action_space_size=${configValues.environment[0].value},
        input_planes=${configValues.environment[1].value},
        height=${configValues.environment[2].value},
        width=${configValues.environment[3].value},
        num_input_moves=${configValues.environment[4].value},
        max_moves=${configValues.environment[5].value},
        num_selfplay_games=${configValues.selfPlay[0].value},
        max_replay_games=${configValues.selfPlay[1].value},
        n_tree_searches=${configValues.selfPlay[2].value},
        training_episodes=${configValues.training[0].value},
        epsilon=${configValues.puct[0].value},
        discount=${configValues.puct[1].value},
        c1=${configValues.puct[2].value},
        c2=${configValues.puct[3].value},
        dirichlet_noise=${configValues.exploration[0].value},
        dirichlet_exploration_factor=${configValues.exploration[1].value},
        batch_size=${configValues.training[1].value},
        info_print_rate=${configValues.logging[0].value},
        training_interval=${configValues.training[2].value},
        learning_rate=${configValues.training[3].value},
        hidden_layer_size=${configValues.neuralNetwork[0].value},
        observation_space_size=${configValues.neuralNetwork[1].value},
        num_training_rolluts=${configValues.training[4].value},
    ):`}
          language="python"
        />
      </div>
    </div>
  );
}
