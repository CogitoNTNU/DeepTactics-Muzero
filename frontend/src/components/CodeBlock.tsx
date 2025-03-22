import { Prism as SyntaxHighlighter } from "react-syntax-highlighter";
import { oneDark } from "react-syntax-highlighter/dist/esm/styles/prism";

type CodeBlockProps = {
  code: string;
  language?: string;
};

export function CodeBlock({ code, language = "javascript" }: CodeBlockProps) {
  return (
    <div className="rounded-lg bg-black p-4">
      <SyntaxHighlighter
        language={language}
        style={oneDark}
        customStyle={{ padding: 0, background: "transparent" }}
      >
        {code}
      </SyntaxHighlighter>
    </div>
  );
}
