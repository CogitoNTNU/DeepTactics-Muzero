import { useState } from "react";
import InputWithTooltip from "./InputWithTooltip";

interface Parameter {
  labelText: string;
  defaultValue: number;
  tooltipMessage: string;
}

interface EnvironmentParametersFormProps {
  title: string;
  parameters: Parameter[];
  onFormValuesChange: (values: Record<string, string>) => void; // Notify parent
}

export default function EnvironmentParametersForm({
  title,
  parameters,
  onFormValuesChange,
}: EnvironmentParametersFormProps) {
  // Initialize state to store input values
  const [,setFormValues] = useState(
    parameters.reduce((acc, param) => {
      acc[param.labelText] = param.defaultValue.toString();
      return acc;
    }, {} as Record<string, string>)
  );

  // Function to update state when an input changes
  const handleValueChange = (labelText: string, value: string) => {
    setFormValues((prevValues) => {
      const newValues = { ...prevValues, [labelText]: value };
      onFormValuesChange(newValues); // Notify parent
      return newValues;
    });
  };

  return (
    <div className="space-y-4 px-2">
      <h2 className="text-lg font-semibold pb-5 underline">{title}</h2>

      {parameters.map((param, index) => (
        <InputWithTooltip
          key={index}
          {...param}
          onValueChange={handleValueChange} // Pass callback to child
        />
      ))}
    </div>
  );
}
