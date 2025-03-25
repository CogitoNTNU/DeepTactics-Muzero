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
}

export default function EnvironmentParametersForm({
  title,
  parameters,
}: EnvironmentParametersFormProps) {
  // Initialize state to store input values
  const [formValues, setFormValues] = useState(
    parameters.reduce((acc, param) => {
      acc[param.labelText] = param.defaultValue.toString();
      return acc;
    }, {} as Record<string, string>)
  );

  // Function to update state when an input changes
  const handleValueChange = (labelText: string, value: string) => {
    setFormValues((prevValues) => ({
      ...prevValues,
      [labelText]: value,
    }));
  };

  return (
    <div className="space-y-4">
      <h2 className="text-lg font-semibold pb-5 underline">{title}</h2>

      {parameters.map((param, index) => (
        <InputWithTooltip
          key={index}
          {...param}
          onValueChange={handleValueChange} // Pass callback to child
        />
      ))}

      {/* Display updated values */}
      <p className="mt-4 font-medium">Current values:</p>
      <ul className="list-disc pl-5">
        {Object.entries(formValues).map(([label, value]) => (
          <li key={label}>
            <strong>{label}:</strong> {value}
          </li>
        ))}
      </ul>
    </div>
  );
}
