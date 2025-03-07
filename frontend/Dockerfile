# Read docs/frontend/dockerfile.md for detailed explanation
FROM node:current AS builder
WORKDIR /app

### Install dependencies and build the project
COPY package*.json ./
RUN npm install
COPY . .
RUN npm run build

### Actual image is created in the following step
FROM nginx:alpine
COPY --from=builder /app/dist /usr/share/nginx/html
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
