import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "../components/ui/card";
import { FaGithub, FaLinkedin } from "react-icons/fa";

interface ProfileCardProps {
  imageSrc: string;
  title: string;
  description: string;
  github?: string;
  linkedin?: string;
}

export default function ProfileCard({ imageSrc, title, description, github, linkedin }: ProfileCardProps) {
  return (
    <Card className="w-80 border shadow-lg rounded-xl">
      <CardHeader className="flex items-center flex-col text-center">
        <img
          src={imageSrc}
          alt={title}
          className="w-50 h-50 rounded-full border-2 border-gray-300 object-cover"
        />
        <CardTitle className="mt-2 text-lg font-semibold text-white">{title}</CardTitle>
        <CardDescription>{description}</CardDescription>
      </CardHeader>
      <CardContent className="flex justify-center gap-4 py-4">
        {github && (
          <a href={github} target="_blank" rel="noopener noreferrer">
            <FaGithub size={24} className="text-gray-700 hover:text-black transition duration-200" />
          </a>
        )}
        {linkedin && (
          <a href={linkedin} target="_blank" rel="noopener noreferrer">
            <FaLinkedin size={24} className="text-blue-700 hover:text-blue-900 transition duration-200" />
          </a>
        )}
      </CardContent>
    </Card>
  );
}