import ImageCard from "../components/ImageCard";

const teamMembers = [
  { imageSrc: "../../public/brage.JPG", title: "Brage", description: "Datateknologi", github: "https://github.com/BrageHK", linkedin: "https://linkedin.com/in/brage" },
  { imageSrc: "../../public/christian.JPG", title: "Christian", description: "Datateknologi", github: "https://github.com/ChristianFredrikJohnsen", linkedin: "https://linkedin.com/in/christian" },
  { imageSrc: "../../public/eirik.JPG", title: "Eirik", description: "Datateknologi", github: "https://github.com/Eiriksol", linkedin: "https://linkedin.com/in/eirik" },
  { imageSrc: "../../public/kristian.JPG", title: "Kristian", description: "Datateknologi", github: "https://github.com/kristiancarlenius", linkedin: "https://linkedin.com/in/kristian" },
  { imageSrc: "../../public/ludvig.JPG", title: "Ludvig", description: "Informatikk", github: "https://github.com/ludvigovrevik", linkedin: "https://linkedin.com/in/ludvig" },
  { imageSrc: "../../public/magnus.JPG", title: "Magnus", description: "Datateknologi", github: undefined, linkedin: "https://linkedin.com/in/magnus" },
  { imageSrc: "../../public/nicolai.JPG", title: "Nicolai", description: "Datateknologi", github: "https://github.com/Nicolai9897", linkedin: "https://linkedin.com/in/nicolai" },
  { imageSrc: "../../public/vegard.JPG", title: "Vegard", description: "Datateknologi", github: "https://github.com/Vegardhgr", linkedin: "https://linkedin.com/in/vegard" },
];

export default function About() {
  return (
    <div className="flex flex-col items-center justify-center py-20 px-4">
      <h1 className="text-4xl font-bold">About us</h1>
      <p className="mt-4 text-lg text-center max-w-2xl">
        We are DeepTactics, creating an implementation of MuZero.
      </p>
      <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-6 mt-10">
        {teamMembers.map((member, index) => (
          <ImageCard key={index} {...member} />
        ))}
      </div>
    </div>
  );
}