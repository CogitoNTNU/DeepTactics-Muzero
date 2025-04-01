"use client"

import { Link } from "react-router-dom";
import React from "react";

const StackingNavbar = () => {
  const leftItems = [
    { to: "/config", label: "Config" },
    { to: "/watch", label: "Watch MuZero" },
  ];

  const rightItems = [
    { to: "/about", label: "About us" },
  ];

  return (
    <nav className="w-full py-6 bg-black/20 backdrop-blur-sm">
      <div className="max-w-7xl mx-auto flex flex-col items-center gap-4">
        {/* Logo and Home link */}
        <Link 
          to="/" 
          className="flex items-center gap-2 mb-4 text-white hover:text-blue-400 transition-colors duration-300"
        >
          <img src="/cogito_white.svg" alt="Cogito" className="h-8 w-8" />
          <span className="text-2xl font-bold">DeepTactics</span>
        </Link>
        
        {/* Navigation items container */}
        <div className="flex justify-center items-center gap-8 w-full">
          {/* Left items */}
          <div className="flex items-center gap-x-4">
            {leftItems.map((item, index) => (
              <NavbarItem key={index} to={item.to}>
                {item.label}
              </NavbarItem>
            ))}
          </div>

          {/* Right items */}
          <div className="flex items-center gap-x-4">
            {rightItems.map((item, index) => (
              <NavbarItem key={index} to={item.to}>
                {item.label}
              </NavbarItem>
            ))}
          </div>
        </div>
      </div>
    </nav>
  );
};

const NavbarItem = ({
  to,
  children,
  style,
}: {
  to: string;
  children: React.ReactNode;
  style?: React.CSSProperties;
}) => {
  return (
    <Link
      className="flex items-center text-sm px-5 py-3 rounded-lg text-white hover:text-blue-400 hover:bg-white/5 transition-all duration-300"
      to={to}
      style={style}
    >
      {children}
    </Link>
  );
};

export { StackingNavbar }; 