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
        action_space_size=${configValues.environment[0].defaultValue},
        input_planes=${configValues.environment[1].defaultValue},
        height=${configValues.environment[2].defaultValue},
        width=${configValues.environment[3].defaultValue},
        num_input_moves=${configValues.environment[4].defaultValue},
        max_moves=${configValues.environment[5].defaultValue},
        num_selfplay_games=${configValues.selfPlay[0].defaultValue},
        max_replay_games=${configValues.selfPlay[1].defaultValue},
        n_tree_searches=${configValues.selfPlay[2].defaultValue},
        training_episodes=${configValues.training[0].defaultValue},
        epsilon=${configValues.puct[0].defaultValue},
        discount=${configValues.puct[1].defaultValue},
        c1=${configValues.puct[2].defaultValue},
        c2=${configValues.puct[3].defaultValue},
        dirichlet_noise=${configValues.exploration[0].defaultValue},
        dirichlet_exploration_factor=${configValues.exploration[1].defaultValue},
        batch_size=${configValues.training[1].defaultValue},
        info_print_rate=${configValues.logging[0].defaultValue},
        training_interval=${configValues.training[2].defaultValue},
        learning_rate=${configValues.training[3].defaultValue},
        hidden_layer_size=${configValues.neuralNetwork[0].defaultValue},
        observation_space_size=${configValues.neuralNetwork[1].defaultValue},
        num_training_rolluts=${configValues.training[4].defaultValue},
    ):`}
          language="python"
        />
      </div>
    </div>
  );
}
