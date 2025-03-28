import { useState } from "react";
import { CodeBlock } from "../components/CodeBlock";
import InputList from "../components/InputList";
import { configValues as initialConfig } from "../config/muzeroParameters";

export default function MuZeroConfig() {
  // State to track updated config values
  const [configValues, setConfigValues] = useState(initialConfig);

  // Function to handle value updates
  const handleConfigChange = (category: string, updatedValues: Record<string, string>) => {
    setConfigValues((prevConfig) => ({
      ...prevConfig,
      [category]: prevConfig[category].map((param) => ({
        ...param,
        defaultValue: updatedValues[param.labelText] || param.defaultValue, // Update only changed values
      })),
    }));
  };

  // Utility function to get the latest value dynamically
  const getValue = (category: string, index: number) => configValues[category]?.[index]?.defaultValue ?? "N/A";

  return (
    <div className="flex flex-col items-center justify-center py-20">
      <h1 className="text-4xl font-bold">Welcome to the config site!</h1>
      <p className="mt-4 text-lg">Finetune the training configs here!</p>
      <div className="pt-10 flex flex-row justify-between w-full">
        {Object.keys(configValues).map((category) => (
          <InputList
            key={category}
            title={category}
            parameters={configValues[category]}
            onFormValuesChange={(updatedValues) => handleConfigChange(category, updatedValues)}
          />
        ))}
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

        # Environment settings
        action_space_size=${getValue("environment", 0)},
        input_planes=${getValue("environment", 1)},
        height=${getValue("environment", 2)},
        width=${getValue("environment", 3)},
        num_input_moves=${getValue("environment", 4)},
        max_moves=${getValue("environment", 5)},

        # Self-play settings
        num_selfplay_games=${getValue("selfPlay", 0)},
        max_replay_games=${getValue("selfPlay", 1)},
        n_tree_searches=${getValue("selfPlay", 2)},

        # Exploration settings
        dirichlet_noise=${getValue("exploration", 0)},
        dirichlet_exploration_factor=${getValue("exploration", 1)},

        # Training settings
        training_episodes=${getValue("training", 0)},
        training_interval=${getValue("training", 1)},
        learning_rate=${getValue("training", 2)},
        batch_size=${getValue("training", 3)},
        momentum=${getValue("training", 4)},
        weight_decay=${getValue("training", 5)},
        lr_decay_steps=${getValue("training", 6)},
        lr_decay_rate=${getValue("training", 7)},
        num_training_rollouts=${getValue("training", 8)},

        # Neural network settings
        hidden_layer_size=${getValue("neuralNetwork", 0)},
        observation_space_size=${getValue("neuralNetwork", 1)},

        # PUCT (policy improvement) settings
        epsilon=${getValue("puct", 0)},
        discount=${getValue("puct", 1)},
        c1=${getValue("puct", 2)},
        c2=${getValue("puct", 3)},

        # Logging settings
        info_print_rate=${getValue("logging", 0)},

    ):`}
          language="python"
        />
      </div>
    </div>
  );
}
