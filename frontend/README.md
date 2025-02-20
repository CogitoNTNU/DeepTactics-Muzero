# React + TypeScript + Vite

- [React + TypeScript + Vite](#react--typescript--vite)
  - [Installation (First-Time Setup)](#installation-first-time-setup)
  - [Running in Development Mode (Hot-Reloading)](#running-in-development-mode-hot-reloading)
  - [Running in Production Mode](#running-in-production-mode)
  - [Expanding the ESLint configuration](#expanding-the-eslint-configuration)

This template provides a minimal setup to get React working in Vite with HMR and some ESLint rules.

Currently, two official plugins are available:

- [@vitejs/plugin-react](https://github.com/vitejs/vite-plugin-react/blob/main/packages/plugin-react/README.md) uses [Babel](https://babeljs.io/) for Fast Refresh
- [@vitejs/plugin-react-swc](https://github.com/vitejs/vite-plugin-react-swc) uses [SWC](https://swc.rs/) for Fast Refresh

## Installation (First-Time Setup)

Before running the project, ensure you have Node.js installed. If not, install it:

- Recommended: Node.js 18+
You can install it using:

```sh
# Using Node Version Manager (NVM)
nvm install 18
nvm use 18
```

Once Node.js is installed, navigate to the `frontend/` folder and run:

```sh
npm install
```

This installs all necessary dependencies, both **regular** and **dev** dependencies.

## Running in Development Mode (Hot-Reloading)

To start the development server (hot-reloading + fast refresh):

```sh
npm run dev
```

- This starts the Vite development server.
- You should see an output like:

```text
VITE v4.3.2  ready in 500ms
Local: http://localhost:5173/
```

✅ Now, open your browser and visit `http://localhost:5173/` to see the app.

## Running in Production Mode

For production, you don’t run a dev server. Instead, you:

1. Build the optimized production version:

```sh
npm run build
```

This creates a fully optimized and minified version inside the dist/ folder.

2. Serve the production build:

```sh
npm run preview
```

This runs a lightweight production server to serve the static files.

✅ Now, visit `http://localhost:4173/` (default Vite preview port) to test the production build.

## Expanding the ESLint configuration

If you are developing a production application, we recommend updating the configuration to enable type aware lint rules:

- Configure the top-level `parserOptions` property like this:

```js
export default tseslint.config({
  languageOptions: {
    // other options...
    parserOptions: {
      project: ['./tsconfig.node.json', './tsconfig.app.json'],
      tsconfigRootDir: import.meta.dirname,
    },
  },
})
```

- Replace `tseslint.configs.recommended` to `tseslint.configs.recommendedTypeChecked` or `tseslint.configs.strictTypeChecked`
- Optionally add `...tseslint.configs.stylisticTypeChecked`
- Install [eslint-plugin-react](https://github.com/jsx-eslint/eslint-plugin-react) and update the config:

```js
// eslint.config.js
import react from 'eslint-plugin-react'

export default tseslint.config({
  // Set the react version
  settings: { react: { version: '18.3' } },
  plugins: {
    // Add the react plugin
    react,
  },
  rules: {
    // other rules...
    // Enable its recommended rules
    ...react.configs.recommended.rules,
    ...react.configs['jsx-runtime'].rules,
  },
})
```
