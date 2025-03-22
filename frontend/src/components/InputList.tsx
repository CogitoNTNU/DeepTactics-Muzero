import InputWithTooltip from "./InputWithTooltip";

interface Parameter {
  labelText: string;
  value: number;
  tooltipMessage: string;
}

interface EnvironmentParametersFormProps {
  title: string;
  parameters: Parameter[];
}

export default function EnvironmentParametersForm({ parameters, title }: EnvironmentParametersFormProps) {
  return (
    <div className="space-y-4">
      <h2 className="text-lg font-semibold pb-5 underline">{title}</h2>
      {parameters.map((param, index) => (
        <InputWithTooltip key={index} {...param} />
      ))}
    </div>
  );
}
