import InputWithTooltip from "./InputWithTooltip";

interface Parameter {
  labelText: string;
  defaultValue: number;
  tooltipMessage: string;
}

interface EnvironmentParametersFormProps {
  parameters: Parameter[];
}

export default function EnvironmentParametersForm({ parameters }: EnvironmentParametersFormProps) {
  return (
    <div className="space-y-4">
      {parameters.map((param, index) => (
        <InputWithTooltip key={index} {...param} />
      ))}
    </div>
  );
}
