import ImageCard from "../components/ImageCard";

const teamMembers = [
  { imageSrc: "/images/placeholder.jpg", title: "Adi", description: "Robotikk", github: "https://github.com/adisinghstudent", linkedin: "https://www.linkedin.com/in/adisinghwork/" },
  { imageSrc: "/images/brage.JPG", title: "Brage", description: "Datateknologi", github: "https://github.com/BrageHK", linkedin: "https://www.linkedin.com/in/brage-kvamme-b33318212/" },
  { imageSrc: "/images/christian.JPG", title: "Christian", description: "Datateknologi", github: "https://github.com/ChristianFredrikJohnsen", linkedin: undefined },
  { imageSrc: "/images/eirik.JPG", title: "Eirik", description: "Datateknologi", github: "https://github.com/Eiriksol", linkedin: "https://no.linkedin.com/in/eirik-solberg-b25bb0252" },
  { imageSrc: "/images/kristian.JPG", title: "Kristian", description: "Datateknologi", github: "https://github.com/kristiancarlenius", linkedin: undefined },
  { imageSrc: "/images/ludvig.JPG", title: "Ludvig", description: "Informatikk", github: "https://github.com/ludvigovrevik", linkedin: "https://no.linkedin.com/in/ludvig-øvrevik-436831214" },
  { imageSrc: "/images/magnus.JPG", title: "Magnus", description: "Dataingeniør", github: undefined, linkedin: "https://no.linkedin.com/in/magnus-wang-wold-bb6b16247" },
  { imageSrc: "/images/nicolai.JPG", title: "Nicolai", description: "Datateknologi", github: "https://github.com/Nicolai9897", linkedin: undefined },
  { imageSrc: "/images/vegard.JPG", title: "Vegard", description: "Datateknologi", github: "https://github.com/Vegardhgr", linkedin: "https://www.linkedin.com/in/vegard-gr%C3%B8der-b04ab1261/" },
];

export default function About() {
  return (
    <div className="flex flex-col items-center justify-center pt-40 pb-20 px-4 text-white">
      <h1 className="text-4xl font-bold">About us</h1>
      <p className="mt-4 text-lg text-center max-w-2xl mb-12">
        We are DeepTactics, creating an implementation of MuZero.
      </p>
      <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-8 max-w-6xl mx-auto w-full">
        {teamMembers.map((member, index) => (
          <div key={index} className="flex justify-center">
            <ImageCard {...member} />
          </div>
        ))}
      </div>
    </div>
  );
}