import ImageCard from "../components/ImageCard";

const teamMembers = [
  { imageSrc: "../../public/adi.JPG", title: "Adi", description: "Automatisering og intelligente systemer", github: "https://github.com/adisinghstudent", linkedin: "https://www.linkedin.com/in/adisinghwork/" },
  { imageSrc: "../../public/brage.JPG", title: "Brage", description: "Datateknologi", github: "https://github.com/BrageHK", linkedin: "https://www.linkedin.com/in/brage-kvamme-b33318212/" },
  { imageSrc: "../../public/christian.JPG", title: "Christian", description: "Datateknologi", github: "https://github.com/ChristianFredrikJohnsen", linkedin: undefined },
  { imageSrc: "../../public/eirik.JPG", title: "Eirik", description: "Datateknologi", github: "https://github.com/Eiriksol", linkedin: "https://no.linkedin.com/in/eirik-solberg-b25bb0252" },
  { imageSrc: "../../public/kristian.JPG", title: "Kristian", description: "Datateknologi", github: "https://github.com/kristiancarlenius", linkedin: undefined },
  { imageSrc: "../../public/ludvig.JPG", title: "Ludvig", description: "Informatikk", github: "https://github.com/ludvigovrevik", linkedin: "https://no.linkedin.com/in/ludvig-øvrevik-436831214" },
  { imageSrc: "../../public/magnus.JPG", title: "Magnus", description: "Dataingeniør", github: undefined, linkedin: "https://no.linkedin.com/in/magnus-wang-wold-bb6b16247" },
  { imageSrc: "../../public/nicolai.JPG", title: "Nicolai", description: "Datateknologi", github: "https://github.com/Nicolai9897", linkedin: undefined },
  { imageSrc: "../../public/vegard.JPG", title: "Vegard", description: "Datateknologi", github: "https://github.com/Vegardhgr", linkedin: "https://www.linkedin.com/in/vegard-gr%C3%B8der-b04ab1261/" },
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