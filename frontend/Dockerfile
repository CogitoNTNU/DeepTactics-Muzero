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

### Copy custom nginx config
COPY nginx.conf /etc/nginx/nginx.conf

EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
