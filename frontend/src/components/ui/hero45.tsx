import { Brain, Cpu, Zap } from "lucide-react";
import { Badge } from "./badge";
import { Separator } from "./separator";

interface Feature {
  icon: React.ReactNode;
  title: string;
  description: string;
}

interface Hero45Props {
  badge?: string;
  heading: string;
  imageSrc?: string;
  imageAlt?: string;
  features?: Feature[];
  children?: React.ReactNode;
}

const Hero45 = ({
  badge = "MuZero AI",
  heading = "Advanced Reinforcement Learning",
  imageSrc = "https://images.unsplash.com/photo-1677442136019-21780ecad995?q=80&w=2532&auto=format&fit=crop",
  imageAlt = "AI Neural Network",
  features = [
    {
      icon: <Brain className="h-auto w-5" />,
      title: "Self-Learning AI",
      description:
        "MuZero learns and masters games without prior knowledge of the rules.",
    },
    {
      icon: <Cpu className="h-auto w-5" />,
      title: "Model-Based Planning",
      description:
        "Advanced planning capabilities through learned dynamics models.",
    },
    {
      icon: <Zap className="h-auto w-5" />,
      title: "State-of-the-Art Performance",
      description:
        "Achieves superhuman performance in various challenging domains.",
    },
  ],
  children,
}: Hero45Props) => {
  return (
    <section className="py-16">
      <div className="container overflow-hidden">
        <div className="mb-20 flex flex-col items-center gap-6 text-center">
          <Badge variant="outline">{badge}</Badge>
          <h1 className="text-4xl font-semibold text-white lg:text-5xl">{heading}</h1>
        </div>
        <div className="relative mx-auto max-w-screen-lg">
          <img
            src={imageSrc}
            alt={imageAlt}
            className="aspect-video max-h-[500px] w-full rounded-xl object-cover"
          />
          <div className="absolute inset-0 bg-gradient-to-t from-background via-transparent to-transparent"></div>
          <div className="absolute -right-28 -top-28 -z-10 aspect-video h-72 w-96 opacity-40 [background-size:12px_12px] [mask-image:radial-gradient(ellipse_50%_50%_at_50%_50%,#000_20%,transparent_100%)] sm:bg-[radial-gradient(hsl(var(--muted-foreground))_1px,transparent_1px)]"></div>
          <div className="absolute -left-28 -top-28 -z-10 aspect-video h-72 w-96 opacity-40 [background-size:12px_12px] [mask-image:radial-gradient(ellipse_50%_50%_at_50%_50%,#000_20%,transparent_100%)] sm:bg-[radial-gradient(hsl(var(--muted-foreground))_1px,transparent_1px)]"></div>
        </div>
        {children}
        {features && features.length > 0 && (
          <div className="mx-auto mt-10 flex max-w-screen-lg flex-col md:flex-row">
            {features.map((feature, index) => (
              <>
                {index > 0 && (
                  <Separator
                    orientation="vertical"
                    className="mx-6 hidden h-auto w-[2px] bg-gradient-to-b from-muted via-transparent to-muted md:block"
                  />
                )}
                <div
                  key={index}
                  className="flex grow basis-0 flex-col rounded-md bg-background/5 p-4 text-white"
                >
                  <div className="mb-6 flex size-10 items-center justify-center rounded-full bg-white/10 drop-shadow-lg">
                    {feature.icon}
                  </div>
                  <h3 className="mb-2 font-semibold">{feature.title}</h3>
                  <p className="text-sm text-muted-foreground">
                    {feature.description}
                  </p>
                </div>
              </>
            ))}
          </div>
        )}
      </div>
    </section>
  );
};

export { Hero45 }; 