"use client"

import { Link } from "react-router-dom";
import React, { useState } from "react";
import { motion } from "framer-motion";

const StackingNavbar = () => {
  const [expanded, setExpanded] = useState(false);

  const leftItems = [
    { to: "/config", label: "Config" },
    { to: "/watch", label: "Watch MuZero" },
  ];

  const rightItems = [
    { to: "/about", label: "About us" },
  ];

  return (
    <nav className="w-full py-6">
      <div className="max-w-7xl mx-auto flex flex-col items-center gap-4">
        {/* Logo and Home link */}
        <Link to="/" className="flex items-center gap-2 mb-4">
          <img src="/cogito_white.svg" alt="Cogito" className="h-8 w-8" />
          <span className="text-2xl font-bold">DeepTactics</span>
        </Link>
        
        {/* Navigation items container */}
        <div className="flex justify-center items-center gap-4 w-full">
          {/* Left items */}
          <div
            className="flex items-center gap-x-2"
            onMouseEnter={() => setExpanded(true)}
            onMouseLeave={() => setExpanded(false)}
          >
            {leftItems.map((item, index) => (
              <StackingNavbarItem
                to={item.to}
                expanded={expanded}
                key={index}
                index={index}
                direction="left"
              >
                {item.label}
              </StackingNavbarItem>
            ))}
          </div>

          {/* Right items */}
          <div
            className="flex items-center gap-x-2"
            onMouseEnter={() => setExpanded(true)}
            onMouseLeave={() => setExpanded(false)}
          >
            {rightItems.map((item, index) => (
              <StackingNavbarItem
                to={item.to}
                expanded={expanded}
                key={index}
                index={index}
                direction="right"
              >
                {item.label}
              </StackingNavbarItem>
            ))}
          </div>
        </div>
      </div>
    </nav>
  );
};

const StackingNavbarItem = ({
  to,
  children,
  style,
  expanded,
  index,
  direction = "left",
}: {
  to: string;
  children: React.ReactNode;
  style?: React.CSSProperties;
  expanded: boolean;
  index: number;
  direction?: "left" | "right";
}) => {
  const initialX = direction === "left" ? -100 * index : 100 * index;
  
  return (
    <motion.div
      initial={{ x: initialX }}
      animate={{ x: expanded ? 0 : initialX }}
      transition={{
        duration: 0.6,
        ease: "circInOut",
        delay: 0.1 * index,
        type: "spring",
      }}
      style={{ zIndex: 100 - index }}
    >
      <Link
        className="flex items-center text-sm px-5 py-3 rounded-3xl bg-[#b0aaaa1a] no-underline text-foreground backdrop-blur-lg hover:bg-black hover:text-white transition-colors duration-300 ease-in-out"
        to={to}
        style={style}
      >
        {children}
      </Link>
    </motion.div>
  );
};

export { StackingNavbar }; 