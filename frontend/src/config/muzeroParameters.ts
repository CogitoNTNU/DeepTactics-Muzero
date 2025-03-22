// Config values for easy access and modification
export const environmentParameters = [
  { labelText: "Action Space Size", value: 2, tooltipMessage: "The number of possible actions in the environment." },
  { labelText: "Input Planes", value: 128, tooltipMessage: "The number of input planes representing the state." },
  { labelText: "Height", value: 96, tooltipMessage: "The pixel height of the input." },
  { labelText: "Width", value: 96, tooltipMessage: "The pixel width of the input." },
  { labelText: "Num Input Moves", value: 32, tooltipMessage: "The number of past moves used as input." },
  { labelText: "Max Moves", value: 50000.0, tooltipMessage: "The maximum number of moves before the game ends." }
];

export const selfPlaySettings = [
  { labelText: "Num Self-Play Games", value: 1000000, tooltipMessage: "Total number of self-play games." },
  { labelText: "Max Replay Games", value: 125000, tooltipMessage: "Size of the replay buffer." },
  { labelText: "N Tree Searches", value: 50, tooltipMessage: "Number of Monte Carlo tree searches per move." }
];

export const explorationSettings = [
  { labelText: "Dirichlet Noise", value: 0.25, tooltipMessage: "The alpha parameter for Dirichlet noise." },
  { labelText: "Dirichlet Exploration Factor", value: 0.25, tooltipMessage: "Exploration factor for Dirichlet noise." }
];

export const trainingParameters = [
  { labelText: "Training Episodes", value: 200, tooltipMessage: "Total number of training episodes." },
  { labelText: "Training Interval", value: 5, tooltipMessage: "Number of steps between training updates." },
  { labelText: "Learning Rate", value: 0.001, tooltipMessage: "Initial learning rate for training." },
  { labelText: "Batch Size", value: 2048, tooltipMessage: "Size of the mini-batch for training." },
  { labelText: "Momentum", value: 0.9, tooltipMessage: "Momentum for the optimizer." },
  { labelText: "Weight Decay", value: 1e-4, tooltipMessage: "Weight decay (L2 regularization)." },
  { labelText: "LR Decay Steps", value: 20, tooltipMessage: "Number of steps before decaying learning rate." },
  { labelText: "LR Decay Rate", value: 0.1, tooltipMessage: "Factor by which the learning rate is decayed." },
  { labelText: "Num Training Rollouts", value: 5, tooltipMessage: "Number of rollouts used during training." }
];

export const neuralNetworkSettings = [
  { labelText: "Hidden Layer Size", value: 32, tooltipMessage: "Size of the hidden layers in the network." },
  { labelText: "Observation Space Size", value: 4, tooltipMessage: "Size of the observation space." }
];

export const puctParameters = [
  { labelText: "C1", value: 1.25, tooltipMessage: "PUCT exploration constant 1." },
  { labelText: "C2", value: 19652, tooltipMessage: "PUCT exploration constant 2." },
  { labelText: "Epsilon", value: 0.001, tooltipMessage: "Epsilon value for exploration." },
  { labelText: "Discount", value: 0.997, tooltipMessage: "Discount factor for future rewards." }
];

export const loggingSettings = [
  { labelText: "Info Print Rate", value: 10, tooltipMessage: "Number of steps between log prints." }
];

// Combine all settings into one object
export const configValues = {
  environment: environmentParameters,
  selfPlay: selfPlaySettings,
  exploration: explorationSettings,
  training: trainingParameters,
  neuralNetwork: neuralNetworkSettings,
  puct: puctParameters,
  logging: loggingSettings,
};
