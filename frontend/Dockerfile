# Use an official Node.js image
FROM node:current AS builder
WORKDIR /app

# Install dependencies and build the project
COPY package*.json ./
RUN npm install
COPY . .
RUN npm run build

# Use a lightweight web server for serving static files
FROM nginx:alpine
COPY --from=builder /app/dist /usr/share/nginx/html
EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]
