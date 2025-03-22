export const environmentParameters = [
  { labelText: "Action Space Size", defaultValue: 2, tooltipMessage: "The number of possible actions in the environment." },
  { labelText: "Input Planes", defaultValue: 128, tooltipMessage: "The number of input planes representing the state." },
  { labelText: "Height", defaultValue: 96, tooltipMessage: "The pixel height of the input." },
  { labelText: "Width", defaultValue: 96, tooltipMessage: "The pixel width of the input." },
  { labelText: "Num Input Moves", defaultValue: 32, tooltipMessage: "The number of past moves used as input." },
  { labelText: "Max Moves", defaultValue: 50000.0, tooltipMessage: "The maximum number of moves before the game ends." }
];

export const selfPlaySettings = [
  { labelText: "Num Self-Play Games", defaultValue: 1000000, tooltipMessage: "Total number of self-play games." },
  { labelText: "Max Replay Games", defaultValue: 125000, tooltipMessage: "Size of the replay buffer." },
  { labelText: "N Tree Searches", defaultValue: 50, tooltipMessage: "Number of Monte Carlo tree searches per move." }
];

export const explorationSettings = [
  { labelText: "Dirichlet Noise", defaultValue: 0.25, tooltipMessage: "The alpha parameter for Dirichlet noise." },
  { labelText: "Dirichlet Exploration Factor", defaultValue: 0.25, tooltipMessage: "Exploration factor for Dirichlet noise." }
];

export const trainingParameters = [
  { labelText: "Training Episodes", defaultValue: 200, tooltipMessage: "Total number of training episodes." },
  { labelText: "Training Interval", defaultValue: 5, tooltipMessage: "Number of steps between training updates." },
  { labelText: "Learning Rate", defaultValue: 0.001, tooltipMessage: "Initial learning rate for training." },
  { labelText: "Batch Size", defaultValue: 2048, tooltipMessage: "Size of the mini-batch for training." },
  { labelText: "Momentum", defaultValue: 0.9, tooltipMessage: "Momentum for the optimizer." },
  { labelText: "Weight Decay", defaultValue: 1e-4, tooltipMessage: "Weight decay (L2 regularization)." },
  { labelText: "LR Decay Steps", defaultValue: 20, tooltipMessage: "Number of steps before decaying learning rate." },
  { labelText: "LR Decay Rate", defaultValue: 0.1, tooltipMessage: "Factor by which the learning rate is decayed." },
  { labelText: "Num Training Rollouts", defaultValue: 5, tooltipMessage: "Number of rollouts used during training." }
];

export const neuralNetworkSettings = [
  { labelText: "Hidden Layer Size", defaultValue: 32, tooltipMessage: "Size of the hidden layers in the network." },
  { labelText: "Observation Space Size", defaultValue: 4, tooltipMessage: "Size of the observation space." }
];

export const puctParameters = [
  { labelText: "C1", defaultValue: 1.25, tooltipMessage: "PUCT exploration constant 1." },
  { labelText: "C2", defaultValue: 19652, tooltipMessage: "PUCT exploration constant 2." },
  { labelText: "Epsilon", defaultValue: 0.001, tooltipMessage: "Epsilon value for exploration." },
  { labelText: "Discount", defaultValue: 0.997, tooltipMessage: "Discount factor for future rewards." }
];

export const loggingSettings = [
  { labelText: "Info Print Rate", defaultValue: 10, tooltipMessage: "Number of steps between log prints." }
];
