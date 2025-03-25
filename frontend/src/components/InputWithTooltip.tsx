import { Label } from "./ui/label";
import { Input } from "./ui/input";
import { Tooltip, TooltipContent, TooltipTrigger } from "./ui/tooltip";
import { Info } from "lucide-react";
import { useState } from "react";

interface InputWithTooltipProps {
  labelText: string;
  defaultValue: number;
  tooltipMessage: string;
  onValueChange: (label: string, value: string) => void; // Callback to send data to parent
}

export default function InputWithTooltip({
  labelText,
  defaultValue,
  tooltipMessage,
  onValueChange,
}: InputWithTooltipProps) {
  const [inputValue, setInputValue] = useState(defaultValue.toString());

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const newValue = e.target.value;
    setInputValue(newValue);
    onValueChange(labelText, newValue); // Send value to parent
  };

  return (
    <div className="flex flex-col space-y-2">
      <div className="flex items-center space-x-2">
        <Label htmlFor="input-field">{labelText}</Label>
        <Tooltip>
          <TooltipTrigger asChild>
            <Info className="w-4 h-4 text-gray-500 cursor-pointer" />
          </TooltipTrigger>
          <TooltipContent>{tooltipMessage}</TooltipContent>
        </Tooltip>
      </div>
      <Input
        id="input-field"
        value={inputValue}
        onChange={handleInputChange}
        type="text"
        placeholder={defaultValue.toString()}
      />
    </div>
  );
}
