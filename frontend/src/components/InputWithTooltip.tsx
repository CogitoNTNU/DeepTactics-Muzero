import { Label } from "./ui/label";
import { Input } from "./ui/input";
import { Tooltip, TooltipContent, TooltipTrigger } from "./ui/tooltip";
import { Info } from "lucide-react";
import { useState } from "react";

interface InputWithTooltipProps {
  labelText: string;
  defaultValue: number;
  tooltipMessage: string;
}

export default function InputWithTooltip({
  labelText,
  defaultValue,
  tooltipMessage,
}: InputWithTooltipProps) {
  const [inputValue, setInputValue] = useState(defaultValue.toString()); // Initialize with defaultValue as a string

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setInputValue(e.target.value); // Extract the value from the event and update state
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
        value={inputValue} // Bind the input value to the state
        onChange={handleInputChange} // Pass the handleInputChange function to update state
        type="text"
        placeholder={defaultValue.toString()}
      />
    </div>
  );
}
