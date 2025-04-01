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
    <Card className="w-80 border border-white/10 bg-white/5 backdrop-blur-sm shadow-lg rounded-xl overflow-hidden">
      <CardHeader className="flex items-center flex-col text-center">
        <img
          src={imageSrc}
          alt={title}
          className="w-40 h-40 rounded-full border-2 border-white/20 object-cover"
        />
        <CardTitle className="mt-4 text-2xl font-semibold text-white">{title}</CardTitle>
        <CardDescription className="text-lg text-gray-300 font-medium">{description}</CardDescription>
      </CardHeader>
      <CardContent className="flex justify-center gap-4 py-4">
        {github && (
          <a href={github} target="_blank" rel="noopener noreferrer">
            <FaGithub size={24} className="text-white/70 hover:text-white transition duration-200" />
          </a>
        )}
        {linkedin && (
          <a href={linkedin} target="_blank" rel="noopener noreferrer">
            <FaLinkedin size={24} className="text-white/70 hover:text-white transition duration-200" />
          </a>
        )}
      </CardContent>
    </Card>
  );
}