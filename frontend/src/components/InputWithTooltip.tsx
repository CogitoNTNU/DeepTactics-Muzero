import { Label } from "./ui/label";
import { Input } from "./ui/input";
import { Tooltip, TooltipContent, TooltipTrigger } from "./ui/tooltip";
import { Info } from "lucide-react";

interface InputWithTooltipProps {
  labelText: string;
  value: number;
  tooltipMessage: string;
}

export default function InputWithTooltip({ labelText, value, tooltipMessage }: InputWithTooltipProps) {
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
      <Input id="input-field" placeholder={value.toString()} defaultValue={value} />
    </div>
  );
}
